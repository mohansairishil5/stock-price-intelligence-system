
"""Search and resolve stock listings and supported market indices."""

import re

import yfinance as yf


# Exact aliases are resolved before fuzzy Yahoo Finance search.
# This prevents a known query from silently resolving to another instrument.
INSTRUMENT_ALIASES = {
    # Indian market indices
    "nifty 50": {
        "name": "NIFTY 50",
        "symbol": "^NSEI",
        "exchange": "NSE",
        "market": "India",
        "type": "INDEX",
    },
    "nifty50": {
        "name": "NIFTY 50",
        "symbol": "^NSEI",
        "exchange": "NSE",
        "market": "India",
        "type": "INDEX",
    },
    "nifty": {
        "name": "NIFTY 50",
        "symbol": "^NSEI",
        "exchange": "NSE",
        "market": "India",
        "type": "INDEX",
    },
    "nifty 50 index": {
        "name": "NIFTY 50",
        "symbol": "^NSEI",
        "exchange": "NSE",
        "market": "India",
        "type": "INDEX",
    },
    "sensex": {
        "name": "BSE SENSEX",
        "symbol": "^BSESN",
        "exchange": "BSE",
        "market": "India",
        "type": "INDEX",
    },
    "bse sensex": {
        "name": "BSE SENSEX",
        "symbol": "^BSESN",
        "exchange": "BSE",
        "market": "India",
        "type": "INDEX",
    },

    # Indian company listings
    "hindustan copper": {
        "name": "Hindustan Copper Limited",
        "symbol": "HINDCOPPER.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },
    "tata motors": {
        "name": "Tata Motors",
        "symbol": "TATAMOTORS.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },
    "reliance": {
        "name": "Reliance Industries Limited",
        "symbol": "RELIANCE.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },
    "reliance industries": {
        "name": "Reliance Industries Limited",
        "symbol": "RELIANCE.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },
    "hdfc bank": {
        "name": "HDFC Bank Limited",
        "symbol": "HDFCBANK.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },
    "tcs": {
        "name": "Tata Consultancy Services Limited",
        "symbol": "TCS.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },
    "tata consultancy services": {
        "name": "Tata Consultancy Services Limited",
        "symbol": "TCS.NS",
        "exchange": "NSE",
        "market": "India",
        "type": "EQUITY",
    },

    # US company aliases
    "apple": {
        "name": "Apple Inc.",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "market": "USA",
        "type": "EQUITY",
    },
    "microsoft": {
        "name": "Microsoft Corporation",
        "symbol": "MSFT",
        "exchange": "NASDAQ",
        "market": "USA",
        "type": "EQUITY",
    },
    "nvidia": {
        "name": "NVIDIA Corporation",
        "symbol": "NVDA",
        "exchange": "NASDAQ",
        "market": "USA",
        "type": "EQUITY",
    },
}

# Supported direct index tickers. These also help classify Yahoo results.
INDEX_TICKERS = {
    "^NSEI": ("NIFTY 50", "NSE", "India"),
    "^BSESN": ("BSE SENSEX", "BSE", "India"),
}


def _normalize_query(query):
    """Normalize a user query for exact alias matching."""
    return re.sub(r"\s+", " ", str(query).strip().casefold())


def _is_indian_instrument(symbol, exchange):
    """Identify likely Indian listings and known Indian indices."""
    symbol = str(symbol).strip().upper()
    exchange = str(exchange).strip().upper()

    return (
        symbol.endswith((".NS", ".BO"))
        or symbol in INDEX_TICKERS
        or "NSE" in exchange
        or "BSE" in exchange
    )


def _make_result(
    name,
    symbol,
    exchange,
    quote_type="EQUITY",
    market="India",
):
    """Build a consistent search-result dictionary."""
    symbol = str(symbol).strip().upper()
    exchange = str(exchange or "Exchange not provided").strip()

    # Known indices have an explicit instrument type and market.
    if symbol in INDEX_TICKERS:
        index_name, index_exchange, index_market = INDEX_TICKERS[symbol]
        name = index_name
        exchange = index_exchange
        quote_type = "INDEX"
        instrument_market = index_market
        matches_market = market == index_market
    else:
        is_indian = _is_indian_instrument(symbol, exchange)
        instrument_market = "India" if is_indian else "USA"

        matches_market = instrument_market == market

    return {
        "name": str(name or symbol),
        "symbol": symbol,
        "exchange": exchange,
        "type": str(quote_type or "Instrument").upper(),
        "market": instrument_market,
        "matches_market": matches_market,
    }


def _alias_result(alias, requested_market):
    """Convert an exact alias to a standard result."""
    return _make_result(
        name=alias["name"],
        symbol=alias["symbol"],
        exchange=alias["exchange"],
        quote_type=alias["type"],
        market=requested_market,
    )


def search_stocks(query, market="India", max_results=15):
    """
    Resolve a company, ticker, or supported index.

    Exact aliases take priority over fuzzy search. Other queries use
    Yahoo Finance search. Search results are not proof that historical
    data is available; the download layer must verify that separately.
    """
    query = str(query).strip()

    if not query:
        return []

    if market not in {"India", "USA"}:
        raise ValueError("Market must be 'India' or 'USA'.")

    normalized_query = _normalize_query(query)

    # 1. Exact known names first: avoid fuzzy misidentification.
    alias = INSTRUMENT_ALIASES.get(normalized_query)

    if alias:
        if alias["market"] != market:
            return []
        return [_alias_result(alias, market)]

    # 2. Direct ticker for supported indices.
    upper_query = query.upper()

    if upper_query in INDEX_TICKERS:
        result = _make_result(
            name=upper_query,
            symbol=upper_query,
            exchange="",
            quote_type="INDEX",
            market=market,
        )
        return [result] if result["matches_market"] else []

    # 3. Search Yahoo Finance for other names and tickers.
    results = []
    seen_symbols = set()
    search_error = None

    try:
        search = yf.Search(
            query,
            max_results=max_results,
            news_count=0,
            enable_fuzzy_query=True,
        )

        for item in search.quotes or []:
            symbol = str(item.get("symbol") or "").strip().upper()

            if not symbol or symbol in seen_symbols:
                continue

            quote_type = str(item.get("quoteType") or "").upper()

            # Include indices; ignore news and unrelated asset types.
            # This project currently supports stocks and market indices.
# Exclude ETFs so an ETF related to a searched stock
# cannot be mistaken for the actual company.
            if quote_type and quote_type not in {"EQUITY", "INDEX"}:
                 continue

            exchange = str(
                item.get("exchDisp") or item.get("exchange") or ""
            ).strip()

            name = (
                item.get("longname")
                or item.get("shortname")
                or item.get("name")
                or symbol
            )

            result = _make_result(
                name=name,
                symbol=symbol,
                exchange=exchange,
                quote_type=quote_type,
                market=market,
            )

            # Keep only results identified as belonging to the chosen
            # market, rather than showing unrelated fallback results.
            if result["matches_market"]:
                results.append(result)
                seen_symbols.add(symbol)

    except Exception as exc:
        search_error = exc

    # 4. Direct ticker fallback. Historical download will verify it.
    looks_like_ticker = (
        "." in query
        or query.startswith("^")
        or (
            query.isupper()
            and " " not in query
            and len(query) <= 15
        )
    )

    if looks_like_ticker and not results:
        symbol = upper_query

        if market == "India" and symbol.endswith((".NS", ".BO")):
            results.append(
                _make_result(
                    name=symbol,
                    symbol=symbol,
                    exchange="NSE" if symbol.endswith(".NS") else "BSE",
                    market=market,
                )
            )
        elif market == "USA" and "." not in symbol and not symbol.startswith("^"):
            results.append(
                _make_result(
                    name=symbol,
                    symbol=symbol,
                    exchange="US exchange not yet verified",
                    market=market,
                )
            )

    # 5. Deduplicate and rank.
    unique_results = {}
    for item in results:
        unique_results.setdefault(item["symbol"], item)

    results = list(unique_results.values())
    results.sort(key=lambda item: item["name"].casefold())

    if search_error and not results:
        raise RuntimeError(
            "Yahoo Finance search is unavailable and no exact alias "
            "or direct ticker fallback matched your query. Please "
            "check the ticker or try again."
        ) from search_error

    return results[:max_results]
