package tools

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// SearchResult is one item returned by a web search.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

// webClient is a shared HTTP client for all web tool requests.
var webClient = &http.Client{Timeout: 15 * time.Second}

// ─── Search ──────────────────────────────────────────────────────────────────

// SearchDDG searches DuckDuckGo for query and returns up to maxResults results.
// It tries the JSON instant-answer API first; if that yields nothing it falls
// back to scraping the lightweight HTML search page.
func SearchDDG(query string, maxResults int) ([]SearchResult, error) {
	if maxResults <= 0 {
		maxResults = 5
	}

	// --- Pass 1: instant-answer JSON API ---
	results, err := searchDDGJSON(query, maxResults)
	if err == nil && len(results) > 0 {
		return results, nil
	}

	// --- Pass 2: lightweight HTML page ---
	return searchDDGHTML(query, maxResults)
}

// searchDDGJSON calls the DuckDuckGo instant-answer API (no API key required).
// Works well for factual / encyclopaedic queries; often empty for news.
func searchDDGJSON(query string, maxResults int) ([]SearchResult, error) {
	apiURL := "https://api.duckduckgo.com/?q=" + url.QueryEscape(query) +
		"&format=json&no_html=1&skip_disambig=1"

	req, err := http.NewRequest(http.MethodGet, apiURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Guido/1.0)")

	resp, err := webClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var ddg struct {
		Heading        string `json:"Heading"`
		Abstract       string `json:"Abstract"`
		AbstractURL    string `json:"AbstractURL"`
		AbstractSource string `json:"AbstractSource"`
		Answer         string `json:"Answer"`
		RelatedTopics  []struct {
			FirstURL string `json:"FirstURL"`
			Text     string `json:"Text"`
		} `json:"RelatedTopics"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ddg); err != nil {
		return nil, err
	}

	var results []SearchResult

	if ddg.Abstract != "" {
		results = append(results, SearchResult{
			Title:   ddg.Heading,
			URL:     ddg.AbstractURL,
			Snippet: ddg.Abstract,
		})
	} else if ddg.Answer != "" {
		results = append(results, SearchResult{
			Title:   "Answer",
			Snippet: ddg.Answer,
		})
	}

	for _, t := range ddg.RelatedTopics {
		if len(results) >= maxResults {
			break
		}
		if t.Text != "" {
			results = append(results, SearchResult{
				URL:     t.FirstURL,
				Snippet: t.Text,
			})
		}
	}

	return results, nil
}

// Regexes for the DuckDuckGo HTML search page.
var (
	reDDGTitle   = regexp.MustCompile(`(?s)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>`)
	reDDGSnippet = regexp.MustCompile(`(?s)<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>`)
	reUDDG       = regexp.MustCompile(`uddg=([^&"]+)`)
)

// searchDDGHTML scrapes the lightweight DuckDuckGo HTML search page.
func searchDDGHTML(query string, maxResults int) ([]SearchResult, error) {
	searchURL := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)

	req, err := http.NewRequest(http.MethodGet, searchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.5")

	resp, err := webClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("DDG HTML search failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return nil, err
	}

	return parseDDGHTML(string(body), maxResults), nil
}

func parseDDGHTML(body string, maxResults int) []SearchResult {
	titleMatches := reDDGTitle.FindAllStringSubmatch(body, maxResults*3)
	snippetMatches := reDDGSnippet.FindAllStringSubmatch(body, maxResults*3)

	var results []SearchResult
	for i, m := range titleMatches {
		if len(results) >= maxResults {
			break
		}

		href := m[1]
		titleHTML := m[2]

		// Decode DDG's redirect URL to get the real destination.
		actualURL := href
		if um := reUDDG.FindStringSubmatch(href); len(um) == 2 {
			if decoded, err := url.QueryUnescape(um[1]); err == nil {
				actualURL = decoded
			}
		}

		// Skip DDG-internal pages.
		if !strings.HasPrefix(actualURL, "http") {
			continue
		}

		snippet := ""
		if i < len(snippetMatches) {
			snippet = cleanText(snippetMatches[i][1])
		}

		results = append(results, SearchResult{
			Title:   cleanText(titleHTML),
			URL:     actualURL,
			Snippet: snippet,
		})
	}
	return results
}

// ─── URL fetch ────────────────────────────────────────────────────────────────

// FetchURL fetches the content at rawURL, strips HTML markup, and returns
// plain text truncated to maxBytes. maxBytes ≤ 0 defaults to 8 000.
func FetchURL(rawURL string, maxBytes int) (string, error) {
	if maxBytes <= 0 {
		maxBytes = 8000
	}

	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; Guido/1.0)")

	resp, err := webClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("server returned %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1024*1024)) // 1 MB cap
	if err != nil {
		return "", err
	}

	text := stripHTML(string(body))
	if len(text) > maxBytes {
		text = text[:maxBytes] + "\n[content truncated]"
	}
	return text, nil
}

// ─── HTML helpers ─────────────────────────────────────────────────────────────

var (
	reScript     = regexp.MustCompile(`(?is)<(script|style)[^>]*>.*?</(script|style)>`)
	reHTMLTag    = regexp.MustCompile(`<[^>]+>`)
	reMultiSpace = regexp.MustCompile(`\s+`)
)

func stripHTML(h string) string {
	h = reScript.ReplaceAllString(h, " ")
	h = reHTMLTag.ReplaceAllString(h, " ")
	return strings.TrimSpace(reMultiSpace.ReplaceAllString(h, " "))
}

func cleanText(s string) string {
	return strings.TrimSpace(reMultiSpace.ReplaceAllString(reHTMLTag.ReplaceAllString(s, " "), " "))
}
