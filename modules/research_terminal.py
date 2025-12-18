"""
Research Terminal Module

Interactive Q&A interface for querying saved research documents (10-K, 10-Q, Transcripts, 8-K)
using Gemini 3.0 Flash Preview.

Migration Notes (Dec 2025):
- Upgraded from google-generativeai (legacy) to google-genai (new SDK)
- Model: gemini-3-flash-preview (smarter than 2.5 Pro, 3x faster)
- Temperature: 1.0 (required for Gemini 3.0 reasoning)
- Thinking Level: HIGH (best accuracy for SEC filing analysis)
- Implicit caching: Automatic after 2,048 token prefix match

Usage:
    from modules.research_terminal import query_research_terminal, get_available_tickers

    # Get tickers with documents
    result = get_available_tickers(db_func)

    # Query documents
    result = query_research_terminal(ticker, question, db_func, gemini_api_key)
"""

import os
import logging
from typing import Dict, List, Any, Callable

LOG = logging.getLogger(__name__)

# Gemini 3.0 Flash Pricing (Dec 2025)
GEMINI_3_INPUT_COST_PER_1M = 0.50   # $0.50 per 1M input tokens
GEMINI_3_OUTPUT_COST_PER_1M = 3.00  # $3.00 per 1M output tokens (includes thinking tokens)

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
# CONTEXT BUILDING
# ------------------------------------------------------------------------------

def build_research_context(ticker: str, db_func: Callable) -> Dict[str, Any]:
    """
    Build context from available research documents for a ticker.

    Uses the same _fetch_available_filings() function as Phase 2 for consistency.

    Args:
        ticker: Stock ticker symbol
        db_func: Database connection function

    Returns:
        Dict with:
            - context_parts: List of formatted document sections
            - sources_used: List of source titles for citation
            - filings: Raw filings dict from Phase 2
    """
    from modules.executive_summary_phase2 import _fetch_available_filings

    filings = _fetch_available_filings(ticker, db_func)

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

    return {
        "context_parts": context_parts,
        "sources_used": sources_used,
        "filings": filings
    }


def build_documents_list(ticker: str, db_func: Callable) -> List[Dict[str, Any]]:
    """
    Build list of available documents for UI display.

    Uses the same _fetch_available_filings() function as Phase 2 for consistency.

    Args:
        ticker: Stock ticker symbol
        db_func: Database connection function

    Returns:
        List of document dicts with type, description, date, and optional item_codes
    """
    from modules.executive_summary_phase2 import _fetch_available_filings

    filings = _fetch_available_filings(ticker, db_func)
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

    return docs


# ------------------------------------------------------------------------------
# MAIN QUERY FUNCTION
# ------------------------------------------------------------------------------

def query_research_terminal(
    ticker: str,
    question: str,
    db_func: Callable,
    gemini_api_key: str
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

    Returns:
        Dict with:
            - answer: AI-generated answer (markdown)
            - sources_used: List of sources cited
            - input_tokens: Token count from API response
            - output_tokens: Token count from API response (includes thinking tokens)
            - cost: Estimated cost in USD
            - cached: Whether implicit caching was used
            - error: Error message if failed (optional)
    """
    # Build context
    context = build_research_context(ticker, db_func)

    if not context["context_parts"]:
        return {"error": f"No research documents found for {ticker}"}

    # Build full context string
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
            # Get all tickers with documents from any of the 4 sources
            cur.execute("""
                SELECT DISTINCT ticker FROM (
                    SELECT ticker FROM sec_filings WHERE filing_type IN ('10-K', '10-Q')
                    UNION
                    SELECT ticker FROM transcript_summaries WHERE report_type = 'transcript'
                    UNION
                    SELECT ticker FROM company_releases WHERE source_type = '8k_exhibit'
                ) AS all_tickers
                ORDER BY ticker
            """)
            tickers = [row['ticker'] for row in cur.fetchall()]

        return {"tickers": tickers}

    except Exception as e:
        LOG.error(f"Failed to get research terminal tickers: {e}")
        return {"error": str(e)}
