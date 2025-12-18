"""
Research Terminal Module

Interactive Q&A interface for querying saved research documents (10-K, 10-Q, Transcripts, 8-K)
using Gemini 2.5 Flash.

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
    Query research documents for a ticker using Gemini 2.5 Flash.

    Args:
        ticker: Stock ticker symbol
        question: User's question
        db_func: Database connection function
        gemini_api_key: Gemini API key

    Returns:
        Dict with:
            - answer: AI-generated answer (markdown)
            - sources_used: List of sources cited
            - input_tokens: Estimated input token count
            - output_tokens: Estimated output token count
            - cost: Estimated cost in USD
            - error: Error message if failed (optional)
    """
    # Build context
    context = build_research_context(ticker, db_func)

    if not context["context_parts"]:
        return {"error": f"No research documents found for {ticker}"}

    # Build full context string
    full_context = "\n\n".join(context["context_parts"])
    sources_used = context["sources_used"]

    # Build the prompt
    system_prompt = RESEARCH_TERMINAL_SYSTEM_PROMPT.format(
        ticker=ticker,
        sources_list=', '.join(sources_used),
        documents_context=full_context
    )

    # Call Gemini 2.5 Flash
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel("gemini-2.5-flash")

        # Estimate input tokens (~4 chars per token)
        input_text = system_prompt + "\n" + question
        estimated_input_tokens = len(input_text) // 4

        response = model.generate_content(
            [{"role": "user", "parts": [{"text": system_prompt + "\n\nQuestion: " + question}]}],
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 16384,
            }
        )

        answer = response.text

        # Estimate output tokens
        estimated_output_tokens = len(answer) // 4

        # Calculate cost (Gemini 2.5 Flash pricing)
        # Input: $0.075 per 1M tokens, Output: $0.30 per 1M tokens
        input_cost = (estimated_input_tokens / 1_000_000) * 0.075
        output_cost = (estimated_output_tokens / 1_000_000) * 0.30
        total_cost = input_cost + output_cost

        return {
            "answer": answer,
            "sources_used": sources_used,
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "cost": total_cost
        }

    except Exception as e:
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

    Args:
        db_func: Database connection function

    Returns:
        Dict with:
            - tickers: List of ticker symbols
            - documents: Dict mapping ticker to list of available documents
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

        # For each ticker, get available documents using Phase 2 logic
        documents = {}
        for ticker in tickers:
            documents[ticker] = build_documents_list(ticker, db_func)

        return {
            "tickers": tickers,
            "documents": documents
        }

    except Exception as e:
        LOG.error(f"Failed to get research terminal tickers: {e}")
        return {"error": str(e)}
