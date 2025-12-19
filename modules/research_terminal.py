"""
Research Terminal Module

Interactive Q&A interface for querying saved research documents (10-K, 10-Q, Transcripts, 8-K)
using Gemini 3.0 Flash Preview or Claude Sonnet 4.5.

Migration Notes (Dec 2025):
- Upgraded from google-generativeai (legacy) to google-genai (new SDK)
- Model: gemini-3-flash-preview (smarter than 2.5 Pro, 3x faster)
- Temperature: 1.0 (required for Gemini 3.0 reasoning)
- Thinking Level: HIGH (best accuracy for SEC filing analysis)
- Implicit caching: Automatic after 2,048 token prefix match
- Added Claude Sonnet 4.5 as alternative model (Dec 2025)

Usage:
    from modules.research_terminal import query_research_terminal, get_available_tickers

    # Get tickers with documents
    result = get_available_tickers(db_func)

    # Query documents (default: Gemini)
    result = query_research_terminal(ticker, question, db_func, gemini_api_key=key)

    # Query with Claude
    result = query_research_terminal(ticker, question, db_func, anthropic_api_key=key, model="claude")
"""

import os
import time
import logging
import requests
from typing import Dict, List, Any, Callable, Optional

LOG = logging.getLogger(__name__)

# Gemini 3.0 Flash Pricing (Dec 2025)
GEMINI_3_INPUT_COST_PER_1M = 0.50   # $0.50 per 1M input tokens
GEMINI_3_OUTPUT_COST_PER_1M = 3.00  # $3.00 per 1M output tokens (includes thinking tokens)

# Claude Sonnet 4.5 Pricing (Dec 2025)
CLAUDE_INPUT_COST_PER_1M = 3.00      # $3.00 per 1M input tokens
CLAUDE_OUTPUT_COST_PER_1M = 15.00    # $15.00 per 1M output tokens
CLAUDE_CACHE_WRITE_PER_1M = 3.75     # $3.75 per 1M cache write tokens
CLAUDE_CACHE_READ_PER_1M = 0.30      # $0.30 per 1M cache read tokens (90% savings)

# ------------------------------------------------------------------------------
# PROMPT TEMPLATE
# ------------------------------------------------------------------------------

RESEARCH_TERMINAL_SYSTEM_PROMPT = """You are a research assistant supporting professional investors conducting fundamental analysis on {ticker}. Your users are experienced - they don't need basics explained.

AVAILABLE DOCUMENTS:
{sources_list}

GUIDELINES:

1. **Synthesize, don't summarize** - Connect information across documents rather than quoting from just one. The value is in the connections.

2. **Be specific** - Use exact figures, dates, and quotes. Vague answers aren't useful.

3. **Note what's changed** - If something evolved between filings or contradicts prior statements, flag it.

4. **Cite inline** - Reference the source naturally (e.g., "per the Q3 transcript" or "the 10-K notes...").

5. **Use markdown** - Format with headers, bullets, and bold for clarity.

If the information isn't in the documents, say so directly rather than speculating.

---

DOCUMENTS:

{documents_context}"""


# ------------------------------------------------------------------------------
# SNAPSHOT HELPER
# ------------------------------------------------------------------------------

def _fetch_snapshot(ticker: str, db_func: Callable) -> Dict[str, Any]:
    """Fetch financial snapshot for a ticker from database."""
    try:
        with db_func() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT ticker, company_name, snapshot_date, current_price, market_cap,
                       shares_outstanding, ebitda_method, snapshot_json
                FROM financial_snapshots
                WHERE ticker = %s
            """, (ticker,))
            row = cur.fetchone()

        if row:
            return {
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'snapshot_date': row['snapshot_date'],
                'current_price': float(row['current_price']) if row['current_price'] else None,
                'market_cap': row['market_cap'],
                'shares_outstanding': row['shares_outstanding'],
                'ebitda_method': row['ebitda_method'],
                'snapshot_json': row['snapshot_json']
            }
        return None
    except Exception as e:
        LOG.warning(f"[{ticker}] Failed to fetch snapshot: {e}")
        return None


def _format_snapshot_for_context(snapshot: Dict) -> str:
    """
    Format snapshot data as readable text for Gemini context.

    Returns a structured text summary of the financial snapshot.
    """
    if not snapshot or not snapshot.get('snapshot_json'):
        return "No snapshot data available."

    data = snapshot['snapshot_json']
    lines = []

    # Header info
    company = data.get('company_name', snapshot.get('ticker', 'N/A'))
    ticker = data.get('ticker', 'N/A')
    sector = data.get('sector', 'N/A')
    industry = data.get('industry', 'N/A')
    price = data.get('current_price')
    mcap = data.get('market_cap')
    shares = data.get('shares_outstanding')

    lines.append(f"COMPANY: {company} ({ticker})")
    lines.append(f"Sector: {sector} | Industry: {industry}")

    price_str = f"${price:.2f}" if price else 'N/A'
    mcap_str = _format_large_num(mcap) if mcap else 'N/A'
    shares_str = _format_large_num(shares) if shares else 'N/A'
    lines.append(f"Current Price: {price_str} | Market Cap: {mcap_str} | Shares Outstanding: {shares_str}")
    lines.append("")

    # Get columns
    annual_cols = data.get('columns', {}).get('annual', [])
    quarterly_cols = data.get('columns', {}).get('quarterly', [])
    metrics = data.get('metrics', {})

    # Format metric data
    metric_defs = [
        ('INCOME STATEMENT', [
            ('Sales', 'Sales ($M)', 'dollar'),
            ('EBITDA', 'EBITDA ($M)', 'dollar'),
            ('EBITDA Margin', 'EBITDA Margin (%)', 'percent'),
            ('Revenue Y/Y', 'Revenue Y/Y (%)', 'percent'),
            ('EPS', 'EPS (Diluted)', 'eps'),
        ]),
        ('CASH FLOW', [
            ('OCF', 'Operating CF ($M)', 'dollar'),
            ('Free Cash Flow', 'Free Cash Flow ($M)', 'dollar'),
        ]),
        ('BALANCE SHEET', [
            ('Gross Debt', 'Gross Debt ($M)', 'dollar'),
            ('Cash', 'Cash ($M)', 'dollar'),
            ('Net Debt', 'Net Debt ($M)', 'dollar'),
            ('Net Leverage', 'Net Leverage (x)', 'multiple'),
        ]),
        ('VALUATION', [
            ('EV/EBITDA', 'EV/EBITDA (x)', 'multiple'),
            ('P/S', 'P/S (x)', 'multiple'),
            ('FCF Yield', 'FCF Yield (%)', 'percent'),
        ]),
    ]

    # Annual data
    if annual_cols:
        lines.append("ANNUAL DATA:")
        header = "Metric".ljust(25) + "".join(str(y).rjust(12) for y in annual_cols)
        lines.append(header)
        lines.append("-" * len(header))

        for section_name, section_metrics in metric_defs:
            for key, label, fmt in section_metrics:
                m = metrics.get(key, {})
                annual_vals = m.get('annual', [])
                vals_str = "".join(_fmt_val(v, fmt).rjust(12) for v in annual_vals[:len(annual_cols)])
                lines.append(f"{label.ljust(25)}{vals_str}")
        lines.append("")

    # Quarterly data (just show key metrics)
    if quarterly_cols:
        lines.append("QUARTERLY DATA (Recent 4 Quarters):")
        recent_quarters = quarterly_cols[-4:] if len(quarterly_cols) >= 4 else quarterly_cols
        header = "Metric".ljust(25) + "".join(q.rjust(12) for q in recent_quarters)
        lines.append(header)
        lines.append("-" * len(header))

        key_metrics = [('Sales', 'Sales ($M)', 'dollar'), ('EBITDA', 'EBITDA ($M)', 'dollar'), ('EPS', 'EPS', 'eps')]
        for key, label, fmt in key_metrics:
            m = metrics.get(key, {})
            quarterly_vals = m.get('quarterly', [])
            recent_vals = quarterly_vals[-4:] if len(quarterly_vals) >= 4 else quarterly_vals
            vals_str = "".join(_fmt_val(v, fmt).rjust(12) for v in recent_vals)
            lines.append(f"{label.ljust(25)}{vals_str}")

    return "\n".join(lines)


def _format_large_num(value) -> str:
    """Format large numbers with T/B/M suffix."""
    if value is None:
        return 'N/A'
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    elif value >= 1e9:
        return f"${value / 1e9:.1f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:,.0f}"


def _fmt_val(value, fmt: str) -> str:
    """Format a metric value for text display."""
    if value is None:
        return '—'
    try:
        if fmt == 'dollar':
            return f"${value:,.0f}"
        elif fmt == 'percent':
            return f"{value:.1f}%"
        elif fmt == 'multiple':
            return f"{value:.1f}x"
        elif fmt == 'eps':
            return f"${value:.2f}"
        return str(value)
    except (ValueError, TypeError):
        return '—'


# ------------------------------------------------------------------------------
# CONTEXT BUILDING
# ------------------------------------------------------------------------------

def build_research_context(ticker: str, db_func: Callable) -> Dict[str, Any]:
    """
    Build context from available research documents for a ticker.

    Uses the same _fetch_available_filings() function as Phase 2 for consistency.
    Also fetches financial snapshot if available.

    Args:
        ticker: Stock ticker symbol
        db_func: Database connection function

    Returns:
        Dict with:
            - context_parts: List of formatted document sections
            - sources_used: List of source titles for citation
            - filings: Raw filings dict from Phase 2 (plus snapshot)
    """
    from modules.executive_summary_phase2 import _fetch_available_filings

    filings = _fetch_available_filings(ticker, db_func)

    # Also fetch snapshot (separate from Phase 2 filings)
    snapshot = _fetch_snapshot(ticker, db_func)
    if snapshot:
        filings['snapshot'] = snapshot

    if not filings:
        return {
            "context_parts": [],
            "sources_used": [],
            "filings": {}
        }

    context_parts = []
    sources_used = []

    # 10-K
    if '10k' in filings and filings['10k'].get('text'):
        title = f"FY{filings['10k']['fiscal_year']} 10-K"
        context_parts.append(f"=== {title} ===\n{filings['10k']['text']}")
        sources_used.append(title)

    # 10-Q
    if '10q' in filings and filings['10q'].get('text'):
        title = f"{filings['10q']['fiscal_quarter']} {filings['10q']['fiscal_year']} 10-Q"
        context_parts.append(f"=== {title} ===\n{filings['10q']['text']}")
        sources_used.append(title)

    # Transcript
    if 'transcript' in filings and filings['transcript'].get('text'):
        title = f"{filings['transcript']['fiscal_quarter']} {filings['transcript']['fiscal_year']} Earnings Call Transcript"
        context_parts.append(f"=== {title} ===\n{filings['transcript']['text']}")
        sources_used.append(title)

    # 8-K filings (post-transcript, filtered by Phase 2 logic - no limit)
    if '8k' in filings:
        for eight_k in filings['8k']:
            if eight_k.get('summary_markdown'):
                filing_date = eight_k['filing_date']
                date_str = filing_date.strftime('%b %d, %Y') if hasattr(filing_date, 'strftime') else str(filing_date)
                title = f"{date_str} 8-K: {eight_k['report_title']}"
                context_parts.append(f"=== {title} ===\n{eight_k['summary_markdown']}")
                sources_used.append(title)

    # Financial Snapshot
    if 'snapshot' in filings and filings['snapshot']:
        snapshot = filings['snapshot']
        snapshot_date = snapshot.get('snapshot_date')
        date_str = snapshot_date.strftime('%b %d, %Y') if hasattr(snapshot_date, 'strftime') else str(snapshot_date)
        title = f"Financial Snapshot ({date_str})"
        # Format snapshot data as readable text for Gemini
        snapshot_text = _format_snapshot_for_context(snapshot)
        context_parts.append(f"=== {title} ===\n{snapshot_text}")
        sources_used.append(title)

    return {
        "context_parts": context_parts,
        "sources_used": sources_used,
        "filings": filings
    }


def build_documents_list(ticker: str, db_func: Callable) -> List[Dict[str, Any]]:
    """
    Build list of available documents for UI display.

    Uses the same _fetch_available_filings() function as Phase 2 for consistency.
    Also fetches financial snapshot if available.

    Args:
        ticker: Stock ticker symbol
        db_func: Database connection function

    Returns:
        List of document dicts with type, description, date, and optional item_codes
    """
    from modules.executive_summary_phase2 import _fetch_available_filings

    filings = _fetch_available_filings(ticker, db_func)

    # Also fetch snapshot (separate from Phase 2 filings)
    snapshot = _fetch_snapshot(ticker, db_func)
    if snapshot:
        filings['snapshot'] = snapshot

    docs = []

    # 10-K
    if '10k' in filings:
        filing_date = filings['10k'].get('filing_date')
        date_str = filing_date.strftime('%b %d, %Y') if hasattr(filing_date, 'strftime') else str(filing_date) if filing_date else 'N/A'
        docs.append({
            'type': '10-K',
            'description': f"FY{filings['10k']['fiscal_year']} 10-K",
            'date': date_str
        })

    # 10-Q
    if '10q' in filings:
        filing_date = filings['10q'].get('filing_date')
        date_str = filing_date.strftime('%b %d, %Y') if hasattr(filing_date, 'strftime') else str(filing_date) if filing_date else 'N/A'
        docs.append({
            'type': '10-Q',
            'description': f"{filings['10q']['fiscal_quarter']} {filings['10q']['fiscal_year']} 10-Q",
            'date': date_str
        })

    # Transcript
    if 'transcript' in filings:
        report_date = filings['transcript'].get('date')
        date_str = report_date.strftime('%b %d, %Y') if hasattr(report_date, 'strftime') else str(report_date) if report_date else 'N/A'
        docs.append({
            'type': 'Transcript',
            'description': f"{filings['transcript']['fiscal_quarter']} {filings['transcript']['fiscal_year']} Earnings Call",
            'date': date_str
        })

    # 8-K filings (post-transcript, filtered by Phase 2 logic - no limit)
    if '8k' in filings:
        for eight_k in filings['8k']:
            filing_date = eight_k.get('filing_date')
            date_str = filing_date.strftime('%b %d, %Y') if hasattr(filing_date, 'strftime') else str(filing_date) if filing_date else 'N/A'
            docs.append({
                'type': '8-K',
                'description': eight_k.get('report_title', 'Unknown'),
                'date': date_str,
                'item_codes': eight_k.get('item_codes')
            })

    # Financial Snapshot
    if 'snapshot' in filings and filings['snapshot']:
        snapshot_date = filings['snapshot'].get('snapshot_date')
        date_str = snapshot_date.strftime('%b %d, %Y') if hasattr(snapshot_date, 'strftime') else str(snapshot_date) if snapshot_date else 'N/A'
        docs.append({
            'type': 'Snapshot',
            'description': f"Financial Snapshot ({date_str})",
            'date': date_str
        })

    return docs


# ------------------------------------------------------------------------------
# GEMINI QUERY FUNCTION
# ------------------------------------------------------------------------------

def _query_research_terminal_gemini(
    ticker: str,
    question: str,
    db_func: Callable,
    gemini_api_key: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Query research documents for a ticker using Gemini 3.0 Flash Preview.

    Uses the new google-genai SDK with:
    - Temperature 1.0 (required for Gemini 3.0 reasoning)
    - Thinking Level HIGH (best accuracy for financial analysis)
    - Implicit caching (automatic after 2,048 token prefix match)

    Args:
        ticker: Stock ticker symbol
        question: User's question
        db_func: Database connection function
        gemini_api_key: Gemini API key
        context: Pre-built context from build_research_context()

    Returns:
        Dict with:
            - answer: AI-generated answer (markdown)
            - sources_used: List of sources cited
            - input_tokens: Token count from API response
            - output_tokens: Token count from API response (includes thinking tokens)
            - cost: Estimated cost in USD
            - cached_tokens: Number of cached tokens
            - model: Model identifier
            - error: Error message if failed (optional)
    """
    # Use pre-built context
    full_context = "\n\n".join(context["context_parts"])
    sources_used = context["sources_used"]

    # Build the system prompt (this is the cacheable prefix)
    system_prompt = RESEARCH_TERMINAL_SYSTEM_PROMPT.format(
        ticker=ticker,
        sources_list=', '.join(sources_used),
        documents_context=full_context
    )

    # Call Gemini 3.0 Flash Preview
    try:
        from google import genai
        from google.genai import types
        from google.genai import errors as genai_errors

        # Initialize client with API key
        client = genai.Client(api_key=gemini_api_key)

        # Build contents with system prompt FIRST (enables implicit caching)
        # The prefix must be identical across requests for caching to work
        contents = [
            types.Part.from_text(text=system_prompt),
            types.Part.from_text(text=f"Question: {question}")
        ]

        # Configure for HIGH thinking (best accuracy) with thoughts hidden from output
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,  # Don't include reasoning in response
                thinking_level=types.ThinkingLevel.HIGH  # Best accuracy for financial analysis
            ),
            temperature=1.0,  # Required for Gemini 3.0 reasoning
            max_output_tokens=16384
        )

        # Make the API call
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=config
        )

        # Extract the answer (with include_thoughts=False, thoughts are omitted from parts)
        answer = "".join([
            part.text for part in response.candidates[0].content.parts
            if not getattr(part, 'thought', False)  # Defensive check in case config changes
        ])

        # Get actual token counts from response metadata
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        # cached_content_token_count may be absent or None on first call (no cache hit)
        cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0

        # Calculate cost (Gemini 3.0 Flash pricing)
        # Cached tokens cost 90% less ($0.05 per 1M vs $0.50)
        uncached_input_tokens = input_tokens - cached_tokens
        input_cost = (uncached_input_tokens / 1_000_000) * GEMINI_3_INPUT_COST_PER_1M
        cache_cost = (cached_tokens / 1_000_000) * 0.05  # Cache read rate
        output_cost = (output_tokens / 1_000_000) * GEMINI_3_OUTPUT_COST_PER_1M
        total_cost = input_cost + cache_cost + output_cost

        LOG.info(f"[{ticker}] Research Terminal: {input_tokens} input ({cached_tokens} cached), {output_tokens} output, ${total_cost:.4f}")

        return {
            "answer": answer,
            "sources_used": sources_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost": total_cost,
            "model": "gemini-3-flash-preview"
        }

    # Specific Gemini 3.0 SDK error handling
    except genai_errors.ClientError as e:
        # 400 errors - config issues, invalid requests, rate limits
        error_str = str(e).lower()
        if 'rate' in error_str or 'quota' in error_str or '429' in error_str:
            LOG.error(f"[{ticker}] Research terminal rate limit: {e}")
            return {"error": "Rate limit exceeded. Please wait a moment and try again."}
        LOG.error(f"[{ticker}] Research terminal client error (400): {e}")
        return {"error": f"Request error: {str(e)}"}

    except genai_errors.ServerError as e:
        # 500/503 errors - Google server issues
        LOG.error(f"[{ticker}] Research terminal server error (5xx): {e}")
        return {"error": "Google AI service temporarily unavailable. Please try again later."}

    except genai_errors.APIError as e:
        # Other API errors
        LOG.error(f"[{ticker}] Research terminal API error: {e}")
        return {"error": f"API error: {str(e)}"}

    except Exception as e:
        # Catch-all for unexpected errors
        LOG.error(f"[{ticker}] Research terminal query failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ------------------------------------------------------------------------------
# CLAUDE QUERY FUNCTION
# ------------------------------------------------------------------------------

def _query_research_terminal_claude(
    ticker: str,
    question: str,
    db_func: Callable,
    anthropic_api_key: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Query research documents for a ticker using Claude Sonnet 4.5.

    Uses direct HTTP API with:
    - Temperature 0.3 (slight creativity for Q&A)
    - Prompt caching via cache_control: ephemeral
    - Retry logic for transient errors

    Args:
        ticker: Stock ticker symbol
        question: User's question
        db_func: Database connection function
        anthropic_api_key: Anthropic API key
        context: Pre-built context from build_research_context()

    Returns:
        Dict with:
            - answer: AI-generated answer (markdown)
            - sources_used: List of sources cited
            - input_tokens: Token count from API response
            - output_tokens: Token count from API response
            - cost: Estimated cost in USD
            - cached_tokens: Number of cached tokens (cache_read_input_tokens)
            - model: Model identifier
            - error: Error message if failed (optional)
    """
    # Use pre-built context
    full_context = "\n\n".join(context["context_parts"])
    sources_used = context["sources_used"]

    # Build the system prompt (cacheable)
    system_prompt = RESEARCH_TERMINAL_SYSTEM_PROMPT.format(
        ticker=ticker,
        sources_list=', '.join(sources_used),
        documents_context=full_context
    )

    # Build request
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    data = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 16384,
        "temperature": 0.3,  # Slight creativity for Q&A
        "system": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}  # Enable prompt caching
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": f"Question: {question}"
            }
        ]
    }

    # Retry logic for transient errors
    max_retries = 2
    response = None
    generation_time_ms = 0

    for attempt in range(max_retries + 1):
        try:
            api_start_time = time.time()
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=120  # 2 minutes
            )
            generation_time_ms = int((time.time() - api_start_time) * 1000)

            # Success - break retry loop
            if response.status_code == 200:
                break

            # Transient errors - retry
            if response.status_code in [429, 500, 503, 529] and attempt < max_retries:
                wait_time = 2 ** attempt
                error_preview = response.text[:200] if response.text else "No details"
                LOG.warning(f"[{ticker}] Research terminal Claude API error {response.status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_preview}")
                LOG.warning(f"[{ticker}] Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Non-retryable error - break
            break

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                LOG.warning(f"[{ticker}] Research terminal Claude timeout (attempt {attempt + 1}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                LOG.error(f"[{ticker}] Research terminal Claude timeout after {max_retries + 1} attempts")
                return {"error": "Request timed out. Please try again."}

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                LOG.warning(f"[{ticker}] Research terminal Claude network error (attempt {attempt + 1}): {e}, retrying...")
                time.sleep(wait_time)
                continue
            else:
                LOG.error(f"[{ticker}] Research terminal Claude network error after {max_retries + 1} attempts: {e}")
                return {"error": f"Network error: {str(e)}"}

    # Check response
    if response is None:
        LOG.error(f"[{ticker}] Research terminal Claude: No response after {max_retries + 1} attempts")
        return {"error": "Failed to get response from Claude API"}

    # Parse response
    if response.status_code == 200:
        result = response.json()
        answer = result.get("content", [{}])[0].get("text", "")

        # Get token counts
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * CLAUDE_INPUT_COST_PER_1M
        output_cost = (output_tokens / 1_000_000) * CLAUDE_OUTPUT_COST_PER_1M
        cache_write_cost = (cache_creation / 1_000_000) * CLAUDE_CACHE_WRITE_PER_1M
        cache_read_cost = (cache_read / 1_000_000) * CLAUDE_CACHE_READ_PER_1M
        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

        LOG.info(f"[{ticker}] Research Terminal (Claude): {input_tokens} input ({cache_read} cached), {output_tokens} output, ${total_cost:.4f}, {generation_time_ms}ms")

        return {
            "answer": answer,
            "sources_used": sources_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cache_read,
            "cost": total_cost,
            "model": "claude-sonnet-4-5-20250929"
        }

    else:
        # Error response
        error_text = response.text[:500] if response.text else "No error details"
        LOG.error(f"[{ticker}] Research terminal Claude API error {response.status_code}: {error_text}")

        if response.status_code == 429:
            return {"error": "Rate limit exceeded. Please wait a moment and try again."}
        elif response.status_code in [500, 503]:
            return {"error": "Claude API temporarily unavailable. Please try again later."}
        else:
            return {"error": f"API error ({response.status_code}): {error_text[:100]}"}


# ------------------------------------------------------------------------------
# MAIN QUERY FUNCTION (ROUTER)
# ------------------------------------------------------------------------------

def query_research_terminal(
    ticker: str,
    question: str,
    db_func: Callable,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    model: str = "gemini"
) -> Dict[str, Any]:
    """
    Query research documents for a ticker using the specified AI model.

    Args:
        ticker: Stock ticker symbol
        question: User's question
        db_func: Database connection function
        gemini_api_key: Gemini API key (required if model="gemini")
        anthropic_api_key: Anthropic API key (required if model="claude")
        model: Model to use - "gemini" (default) or "claude"

    Returns:
        Dict with:
            - answer: AI-generated answer (markdown)
            - sources_used: List of sources cited
            - input_tokens: Token count from API response
            - output_tokens: Token count from API response
            - cost: Estimated cost in USD
            - cached_tokens: Number of cached tokens
            - model: Model identifier
            - error: Error message if failed (optional)
    """
    # Validate model choice
    model = model.lower().strip()
    if model not in ["gemini", "claude"]:
        return {"error": f"Invalid model: {model}. Use 'gemini' or 'claude'."}

    # Validate API key for selected model
    if model == "gemini" and not gemini_api_key:
        return {"error": "Gemini API key not configured"}
    if model == "claude" and not anthropic_api_key:
        return {"error": "Anthropic API key not configured"}

    # Build context once (shared by both models)
    context = build_research_context(ticker, db_func)

    if not context["context_parts"]:
        return {"error": f"No research documents found for {ticker}"}

    # Route to appropriate model
    if model == "claude":
        return _query_research_terminal_claude(ticker, question, db_func, anthropic_api_key, context)
    else:
        return _query_research_terminal_gemini(ticker, question, db_func, gemini_api_key, context)


# ------------------------------------------------------------------------------
# TICKER LISTING
# ------------------------------------------------------------------------------

def get_available_tickers(db_func: Callable) -> Dict[str, Any]:
    """
    Get all tickers that have at least one research document.

    Fast query - only returns ticker list. Documents fetched separately on selection.

    Args:
        db_func: Database connection function

    Returns:
        Dict with:
            - tickers: List of ticker symbols
            - error: Error message if failed (optional)
    """
    try:
        with db_func() as conn, conn.cursor() as cur:
            # Get all tickers with documents from any of the 5 sources
            cur.execute("""
                SELECT DISTINCT ticker FROM (
                    SELECT ticker FROM sec_filings WHERE filing_type IN ('10-K', '10-Q')
                    UNION
                    SELECT ticker FROM transcript_summaries WHERE report_type = 'transcript'
                    UNION
                    SELECT ticker FROM company_releases WHERE source_type = '8k_exhibit'
                    UNION
                    SELECT ticker FROM financial_snapshots
                ) AS all_tickers
                ORDER BY ticker
            """)
            tickers = [row['ticker'] for row in cur.fetchall()]

        return {"tickers": tickers}

    except Exception as e:
        LOG.error(f"Failed to get research terminal tickers: {e}")
        return {"error": str(e)}
