"""
Financial Snapshot Module

Fetches financial data from FMP and processes it into a structured snapshot
for display in the admin dashboard.

Usage:
    from modules.financial_snapshot import FinancialSnapshotAnalyzer

    analyzer = FinancialSnapshotAnalyzer(fmp_api_key)
    result = analyzer.get_financial_snapshot("AAPL")
"""

import logging
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

LOG = logging.getLogger(__name__)


def _convert_to_native(value):
    """
    Convert numpy types to native Python types for JSON serialization.
    FastAPI's jsonable_encoder fails on numpy.int64, numpy.float64, etc.
    """
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _convert_list(values: List) -> List:
    """Convert all values in a list to native Python types"""
    return [_convert_to_native(v) for v in values]

# ------------------------------------------------------------------------------
# FMP CLIENT
# ------------------------------------------------------------------------------

class FMPClientError(Exception):
    """Base exception for FMP API errors"""
    pass


class TickerNotFoundError(FMPClientError):
    """Raised when ticker is not found"""
    pass


class NoFinancialDataError(FMPClientError):
    """Raised when no financial data is available"""
    pass


class FinancialSnapshotFetcher:
    """
    FMP API client for fetching financial data.
    Designed to be simple and stateless - no caching at this layer.
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"
    TIMEOUT = 30

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """Make API request with error handling"""
        url = f"{self.BASE_URL}{endpoint}"

        if params is None:
            params = {}
        params['apikey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=self.TIMEOUT)

            if response.status_code == 404:
                raise TickerNotFoundError(f"Endpoint not found: {endpoint}")

            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if isinstance(data, dict) and 'Error Message' in data:
                error_msg = data['Error Message']
                if 'not found' in error_msg.lower():
                    raise TickerNotFoundError(error_msg)
                raise FMPClientError(error_msg)

            return data

        except requests.exceptions.Timeout:
            raise FMPClientError(f"Request timeout for {endpoint}")
        except requests.exceptions.RequestException as e:
            raise FMPClientError(f"Request failed: {str(e)}")

    def fetch_income_statement(self, ticker: str, period: str = 'annual', limit: int = 10) -> pd.DataFrame:
        """Fetch income statement data"""
        endpoint = f"/income-statement/{ticker.upper()}"
        data = self._make_request(endpoint, {'period': period, 'limit': limit})

        if not data:
            raise NoFinancialDataError(f"No income statement data for {ticker}")

        return pd.DataFrame(data)

    def fetch_balance_sheet(self, ticker: str, period: str = 'annual', limit: int = 10) -> pd.DataFrame:
        """Fetch balance sheet data"""
        endpoint = f"/balance-sheet-statement/{ticker.upper()}"
        data = self._make_request(endpoint, {'period': period, 'limit': limit})

        if not data:
            raise NoFinancialDataError(f"No balance sheet data for {ticker}")

        return pd.DataFrame(data)

    def fetch_cash_flow(self, ticker: str, period: str = 'annual', limit: int = 10) -> pd.DataFrame:
        """Fetch cash flow statement data"""
        endpoint = f"/cash-flow-statement/{ticker.upper()}"
        data = self._make_request(endpoint, {'period': period, 'limit': limit})

        if not data:
            raise NoFinancialDataError(f"No cash flow data for {ticker}")

        return pd.DataFrame(data)

    def fetch_key_metrics(self, ticker: str, period: str = 'annual', limit: int = 10) -> pd.DataFrame:
        """Fetch key metrics data"""
        endpoint = f"/key-metrics/{ticker.upper()}"
        data = self._make_request(endpoint, {'period': period, 'limit': limit})

        # Key metrics might not be available for all companies
        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data)

    def fetch_profile(self, ticker: str) -> Dict:
        """Fetch company profile"""
        endpoint = f"/profile/{ticker.upper()}"
        data = self._make_request(endpoint)

        if not data:
            raise TickerNotFoundError(f"Ticker {ticker} not found")

        return data[0] if isinstance(data, list) else data

    def fetch_quote(self, ticker: str) -> Dict:
        """Fetch current quote data"""
        endpoint = f"/quote/{ticker.upper()}"
        data = self._make_request(endpoint)

        if not data:
            raise TickerNotFoundError(f"Quote not found for {ticker}")

        return data[0] if isinstance(data, list) else data

    def fetch_shares_float(self, ticker: str) -> Optional[Dict]:
        """
        Fetch current shares outstanding from v4 shares_float endpoint.
        Returns None if endpoint fails (graceful degradation).
        """
        try:
            url = "https://financialmodelingprep.com/api/v4/shares_float"
            params = {'symbol': ticker.upper(), 'apikey': self.api_key}
            response = requests.get(url, params=params, timeout=self.TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            LOG.warning(f"[SNAPSHOT] Could not fetch shares float for {ticker}: {e}")
            return None


# ------------------------------------------------------------------------------
# FINANCIAL ANALYZER
# ------------------------------------------------------------------------------

class FinancialSnapshotAnalyzer:
    """
    Analyzes financial data and prepares structured snapshot for display.

    Time Period Logic:
    - Annual: Last 5 full calendar years (e.g., 2020-2024)
    - Quarterly: Fixed 12-quarter grid spanning 3 years, ending at Q4 of current year
      - Unreported quarters show as empty (not shifted)
    """

    ANNUAL_YEARS = 5
    QUARTERLY_YEARS = 3  # 12 quarters = 3 years

    def __init__(self, api_key: str):
        self.fetcher = FinancialSnapshotFetcher(api_key)

    def get_financial_snapshot(self, ticker: str) -> Dict:
        """
        Main entry point - fetch and process all financial data for a ticker.

        Returns:
            {
                'ticker': str,
                'company_name': str,
                'sector': str,
                'industry': str,
                'current_price': float,
                'market_cap': float,
                'ebitda_method': str,
                'generated_at': str,
                'columns': {
                    'annual': [2020, 2021, 2022, 2023, 2024],
                    'quarterly': ['Q1-23', 'Q2-23', ..., 'Q4-25']
                },
                'metrics': {
                    'Sales': {'annual': [...], 'quarterly': [...]},
                    'EBITDA': {'annual': [...], 'quarterly': [...]},
                    ...
                }
            }
        """
        ticker = ticker.upper().strip()
        LOG.info(f"[SNAPSHOT] Generating financial snapshot for {ticker}")

        # Fetch company info
        profile = self.fetcher.fetch_profile(ticker)
        quote = self.fetcher.fetch_quote(ticker)
        shares_float = self.fetcher.fetch_shares_float(ticker)

        company_name = profile.get('companyName', ticker)
        sector = profile.get('sector', 'N/A')
        industry = profile.get('industry', 'N/A')
        current_price = quote.get('price', 0)
        market_cap = quote.get('marketCap', 0)

        # Current shares outstanding (from shares_float endpoint)
        shares_outstanding = None
        float_shares = None
        if shares_float:
            shares_outstanding = shares_float.get('outstandingShares')
            float_shares = shares_float.get('floatShares')

        # Get time periods
        annual_years = self._get_annual_years()
        quarterly_labels = self._get_quarterly_labels()

        # Fetch financial data
        # Annual: fetch enough to cover 5 years + 1 for Y/Y calculations
        income_annual = self.fetcher.fetch_income_statement(ticker, 'annual', self.ANNUAL_YEARS + 1)
        balance_annual = self.fetcher.fetch_balance_sheet(ticker, 'annual', self.ANNUAL_YEARS + 1)
        cashflow_annual = self.fetcher.fetch_cash_flow(ticker, 'annual', self.ANNUAL_YEARS + 1)

        try:
            metrics_annual = self.fetcher.fetch_key_metrics(ticker, 'annual', self.ANNUAL_YEARS + 1)
        except Exception:
            metrics_annual = pd.DataFrame()

        # Quarterly: fetch 16 quarters to ensure coverage + Y/Y calculations
        income_quarterly = self.fetcher.fetch_income_statement(ticker, 'quarter', 16)
        balance_quarterly = self.fetcher.fetch_balance_sheet(ticker, 'quarter', 16)
        cashflow_quarterly = self.fetcher.fetch_cash_flow(ticker, 'quarter', 16)

        try:
            metrics_quarterly = self.fetcher.fetch_key_metrics(ticker, 'quarter', 16)
        except Exception:
            metrics_quarterly = pd.DataFrame()

        # Process data
        annual_data, ebitda_method = self._process_annual_data(
            income_annual, balance_annual, cashflow_annual, metrics_annual,
            annual_years, quote
        )

        quarterly_data = self._process_quarterly_data(
            income_quarterly, balance_quarterly, cashflow_quarterly, metrics_quarterly,
            quarterly_labels, quote
        )

        # Combine into final structure (convert numpy types to native Python for JSON)
        metrics = {}
        for metric_name in annual_data.keys():
            metrics[metric_name] = {
                'annual': _convert_list(annual_data[metric_name]),
                'quarterly': _convert_list(quarterly_data.get(metric_name, [None] * len(quarterly_labels)))
            }

        return {
            'ticker': ticker,
            'company_name': company_name,
            'sector': sector,
            'industry': industry,
            'current_price': _convert_to_native(current_price),
            'market_cap': _convert_to_native(market_cap),
            'shares_outstanding': _convert_to_native(shares_outstanding),
            'float_shares': _convert_to_native(float_shares),
            'ebitda_method': ebitda_method,
            'generated_at': datetime.now().isoformat(),
            'columns': {
                'annual': annual_years,
                'quarterly': quarterly_labels
            },
            'metrics': metrics
        }

    def _get_annual_years(self) -> List[int]:
        """
        Get last N full calendar years.
        Example (Dec 2025): [2020, 2021, 2022, 2023, 2024]
        """
        current_year = datetime.now().year
        # Last 5 FULL years (not including current year which is incomplete)
        return [current_year - self.ANNUAL_YEARS + i for i in range(self.ANNUAL_YEARS)]

    def _get_quarterly_labels(self) -> List[str]:
        """
        Get fixed 12-quarter grid ending at Q4 of current year.
        Example (Dec 2025): ['Q1-23', 'Q2-23', 'Q3-23', 'Q4-23', 'Q1-24', ..., 'Q4-25']
        """
        current_year = datetime.now().year
        start_year = current_year - self.QUARTERLY_YEARS + 1  # 3 years back + current

        quarters = []
        for year in range(start_year, current_year + 1):
            year_short = str(year)[2:]
            for q in range(1, 5):
                quarters.append(f"Q{q}-{year_short}")

        return quarters

    def _date_to_quarter_label(self, date_str: str) -> Optional[str]:
        """Convert date string to quarter label (e.g., '2024-09-30' -> 'Q3-24')"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            quarter = (date_obj.month - 1) // 3 + 1
            year_short = str(date_obj.year)[2:]
            return f"Q{quarter}-{year_short}"
        except Exception:
            return None

    def _get_metric_from_row(self, row: pd.Series, field_names: List[str]) -> Optional[float]:
        """Try to get a metric from a row using multiple possible field names"""
        if row is None or (isinstance(row, pd.Series) and row.empty):
            return None

        for field in field_names:
            if field in row and pd.notna(row[field]):
                return row[field]
        return None

    def _get_ebitda_with_source(self, income_row: pd.Series,
                                 metrics_row: Optional[pd.Series] = None) -> Tuple[Optional[float], str]:
        """
        Get EBITDA using priority hierarchy:
        1. Adjusted EBITDA (from key metrics)
        2. EBITDA (from income statement)
        3. Calculated (Operating Income + D&A)
        """
        # Try Adjusted EBITDA from key metrics
        if metrics_row is not None and not metrics_row.empty:
            if 'adjustedEBITDA' in metrics_row and pd.notna(metrics_row.get('adjustedEBITDA')):
                return (metrics_row['adjustedEBITDA'], 'Adjusted EBITDA')

        # Try EBITDA from income statement
        if income_row is not None and not income_row.empty:
            if 'ebitda' in income_row and pd.notna(income_row.get('ebitda')):
                return (income_row['ebitda'], 'EBITDA')

            # Calculate from Operating Income + D&A
            operating_income = income_row.get('operatingIncome')
            depreciation = income_row.get('depreciationAndAmortization')

            if pd.notna(operating_income) and pd.notna(depreciation):
                return (operating_income + depreciation, 'Calculated (OpInc + D&A)')

        return (None, 'N/A')

    def _calculate_yoy_growth(self, current: Optional[float],
                               previous: Optional[float]) -> Optional[float]:
        """Calculate year-over-year growth percentage"""
        if current is None or previous is None or pd.isna(current) or pd.isna(previous):
            return None
        if previous == 0:
            return None
        return ((current - previous) / abs(previous)) * 100

    def _get_row_by_year(self, df: pd.DataFrame, year: int) -> Optional[pd.Series]:
        """Get row from DataFrame matching a specific year"""
        if df is None or df.empty:
            return None

        # Try calendarYear field first
        if 'calendarYear' in df.columns:
            # Convert to int for comparison
            df_copy = df.copy()
            df_copy['calendarYear'] = pd.to_numeric(df_copy['calendarYear'], errors='coerce')
            matching = df_copy[df_copy['calendarYear'] == year]
            if not matching.empty:
                return matching.iloc[0]

        # Try parsing date field
        if 'date' in df.columns:
            for _, row in df.iterrows():
                date_str = row.get('date')
                if isinstance(date_str, str) and date_str.startswith(str(year)):
                    return row

        return None

    def _build_quarter_mapping(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Build mapping from quarter label to data row"""
        mapping = {}

        if df is None or df.empty or 'date' not in df.columns:
            return mapping

        for _, row in df.iterrows():
            date_str = row.get('date')
            if date_str and isinstance(date_str, str):
                quarter_label = self._date_to_quarter_label(date_str)
                if quarter_label:
                    mapping[quarter_label] = row

        return mapping

    def _process_annual_data(self, income_df: pd.DataFrame, balance_df: pd.DataFrame,
                              cashflow_df: pd.DataFrame, metrics_df: pd.DataFrame,
                              years: List[int], quote: Dict) -> Tuple[Dict, str]:
        """Process annual financial data and calculate metrics"""

        result = {
            'Sales': [],
            'EBITDA': [],
            'EBITDA Margin': [],
            'Revenue Y/Y': [],
            'EBITDA Y/Y': [],
            'EPS': [],
            'Shares Outstanding': [],
            'OCF': [],
            'CapEx': [],
            'Free Cash Flow': [],
            'Gross Debt': [],
            'Cash': [],
            'Net Debt': [],
            'Net Leverage': [],
            'Market Cap': [],
            'EV': [],
            'EV/EBITDA': [],
            'P/S': [],
            'FCF Yield': [],
            'Dividend Yield': [],
        }

        ebitda_methods = []

        for year in years:
            income_row = self._get_row_by_year(income_df, year)
            balance_row = self._get_row_by_year(balance_df, year)
            cashflow_row = self._get_row_by_year(cashflow_df, year)
            metrics_row = self._get_row_by_year(metrics_df, year) if not metrics_df.empty else None

            # Previous year for Y/Y
            prev_income = self._get_row_by_year(income_df, year - 1)

            if income_row is None:
                for key in result:
                    result[key].append(None)
                continue

            # Sales
            revenue = self._get_metric_from_row(income_row, ['revenue', 'totalRevenue'])
            result['Sales'].append(revenue / 1_000_000 if revenue else None)

            # EBITDA
            ebitda, ebitda_method = self._get_ebitda_with_source(income_row, metrics_row)
            ebitda_methods.append(ebitda_method)
            result['EBITDA'].append(ebitda / 1_000_000 if ebitda else None)

            # EBITDA Margin
            if revenue and ebitda:
                result['EBITDA Margin'].append((ebitda / revenue) * 100)
            else:
                result['EBITDA Margin'].append(None)

            # Revenue Y/Y
            if prev_income is not None:
                prev_revenue = self._get_metric_from_row(prev_income, ['revenue', 'totalRevenue'])
                result['Revenue Y/Y'].append(self._calculate_yoy_growth(revenue, prev_revenue))
            else:
                result['Revenue Y/Y'].append(None)

            # EBITDA Y/Y
            if prev_income is not None:
                prev_ebitda, _ = self._get_ebitda_with_source(prev_income)
                result['EBITDA Y/Y'].append(self._calculate_yoy_growth(ebitda, prev_ebitda))
            else:
                result['EBITDA Y/Y'].append(None)

            # EPS (Diluted)
            eps = self._get_metric_from_row(income_row, ['epsdiluted', 'eps'])
            result['EPS'].append(eps)

            # Shares Outstanding (Diluted) - in millions
            shares = self._get_metric_from_row(income_row, ['weightedAverageShsOutDil', 'weightedAverageShsOut'])
            result['Shares Outstanding'].append(shares / 1_000_000 if shares else None)

            # Cash flow metrics
            if cashflow_row is not None:
                ocf = self._get_metric_from_row(cashflow_row, ['operatingCashFlow', 'netCashProvidedByOperatingActivities'])
                capex = self._get_metric_from_row(cashflow_row, ['capitalExpenditure', 'capitalExpenditures'])

                result['OCF'].append(ocf / 1_000_000 if ocf else None)
                result['CapEx'].append(capex / 1_000_000 if capex else None)

                if ocf is not None and capex is not None:
                    fcf = ocf - abs(capex)
                    result['Free Cash Flow'].append(fcf / 1_000_000)
                else:
                    result['Free Cash Flow'].append(None)
            else:
                result['OCF'].append(None)
                result['CapEx'].append(None)
                result['Free Cash Flow'].append(None)

            # Balance sheet metrics
            if balance_row is not None:
                total_debt = self._get_metric_from_row(balance_row, ['totalDebt', 'longTermDebt'])
                cash = self._get_metric_from_row(balance_row, ['cashAndCashEquivalents', 'cash'])

                result['Gross Debt'].append(total_debt / 1_000_000 if total_debt else None)
                result['Cash'].append(cash / 1_000_000 if cash else None)

                if total_debt is not None and cash is not None:
                    net_debt = total_debt - cash
                    result['Net Debt'].append(net_debt / 1_000_000)

                    if ebitda and ebitda != 0:
                        result['Net Leverage'].append(net_debt / ebitda)
                    else:
                        result['Net Leverage'].append(None)
                else:
                    result['Net Debt'].append(None)
                    result['Net Leverage'].append(None)
            else:
                result['Gross Debt'].append(None)
                result['Cash'].append(None)
                result['Net Debt'].append(None)
                result['Net Leverage'].append(None)

            # Market metrics
            if metrics_row is not None and not metrics_row.empty and pd.notna(metrics_row.get('marketCap')):
                market_cap = metrics_row['marketCap']
            else:
                market_cap = quote.get('marketCap')

            result['Market Cap'].append(market_cap / 1_000_000 if market_cap else None)

            # EV
            net_debt_val = result['Net Debt'][-1]
            market_cap_val = result['Market Cap'][-1]

            if market_cap_val and net_debt_val is not None:
                ev = market_cap_val + net_debt_val
                result['EV'].append(ev)

                ebitda_val = result['EBITDA'][-1]
                if ebitda_val and ebitda_val != 0:
                    result['EV/EBITDA'].append(ev / ebitda_val)
                else:
                    result['EV/EBITDA'].append(None)
            else:
                result['EV'].append(None)
                result['EV/EBITDA'].append(None)

            # P/S
            revenue_val = result['Sales'][-1]
            if market_cap_val and revenue_val and revenue_val != 0:
                result['P/S'].append(market_cap_val / revenue_val)
            else:
                result['P/S'].append(None)

            # FCF Yield
            fcf_val = result['Free Cash Flow'][-1]
            if market_cap_val and fcf_val is not None and market_cap_val != 0:
                result['FCF Yield'].append((fcf_val / market_cap_val) * 100)
            else:
                result['FCF Yield'].append(None)

            # Dividend Yield
            if metrics_row is not None and not metrics_row.empty:
                div_yield = self._get_metric_from_row(metrics_row, ['dividendYield'])
                if div_yield is not None:
                    result['Dividend Yield'].append(div_yield * 100)  # Convert to percentage
                else:
                    result['Dividend Yield'].append(None)
            else:
                result['Dividend Yield'].append(None)

        # Determine most common EBITDA method
        valid_methods = [m for m in ebitda_methods if m != 'N/A']
        if valid_methods:
            final_method = max(set(valid_methods), key=valid_methods.count)
        else:
            final_method = 'N/A'

        return result, final_method

    def _process_quarterly_data(self, income_df: pd.DataFrame, balance_df: pd.DataFrame,
                                 cashflow_df: pd.DataFrame, metrics_df: pd.DataFrame,
                                 quarter_labels: List[str], quote: Dict) -> Dict:
        """Process quarterly financial data"""

        result = {
            'Sales': [],
            'EBITDA': [],
            'EBITDA Margin': [],
            'Revenue Y/Y': [],
            'EBITDA Y/Y': [],
            'EPS': [],
            'Shares Outstanding': [],
            'OCF': [],
            'CapEx': [],
            'Free Cash Flow': [],
            'Gross Debt': [],
            'Cash': [],
            'Net Debt': [],
            'Net Leverage': [],
            'Market Cap': [],
            'EV': [],
            'EV/EBITDA': [],
            'P/S': [],
            'FCF Yield': [],
            'Dividend Yield': [],
        }

        # Build quarter mappings
        income_map = self._build_quarter_mapping(income_df)
        balance_map = self._build_quarter_mapping(balance_df)
        cashflow_map = self._build_quarter_mapping(cashflow_df)
        metrics_map = self._build_quarter_mapping(metrics_df) if not metrics_df.empty else {}

        for i, quarter_label in enumerate(quarter_labels):
            income_row = income_map.get(quarter_label)
            balance_row = balance_map.get(quarter_label)
            cashflow_row = cashflow_map.get(quarter_label)
            metrics_row = metrics_map.get(quarter_label)

            # Previous year same quarter for Y/Y (4 quarters back)
            prev_quarter_idx = i - 4
            prev_quarter_label = quarter_labels[prev_quarter_idx] if prev_quarter_idx >= 0 else None
            prev_income = income_map.get(prev_quarter_label) if prev_quarter_label else None

            if income_row is None:
                for key in result:
                    result[key].append(None)
                continue

            # Sales
            revenue = self._get_metric_from_row(income_row, ['revenue', 'totalRevenue'])
            result['Sales'].append(revenue / 1_000_000 if revenue else None)

            # EBITDA
            ebitda, _ = self._get_ebitda_with_source(income_row, metrics_row)
            result['EBITDA'].append(ebitda / 1_000_000 if ebitda else None)

            # EBITDA Margin
            if revenue and ebitda:
                result['EBITDA Margin'].append((ebitda / revenue) * 100)
            else:
                result['EBITDA Margin'].append(None)

            # Revenue Y/Y
            if prev_income is not None:
                prev_revenue = self._get_metric_from_row(prev_income, ['revenue', 'totalRevenue'])
                result['Revenue Y/Y'].append(self._calculate_yoy_growth(revenue, prev_revenue))
            else:
                result['Revenue Y/Y'].append(None)

            # EBITDA Y/Y
            if prev_income is not None:
                prev_ebitda, _ = self._get_ebitda_with_source(prev_income)
                result['EBITDA Y/Y'].append(self._calculate_yoy_growth(ebitda, prev_ebitda))
            else:
                result['EBITDA Y/Y'].append(None)

            # EPS (Diluted)
            eps = self._get_metric_from_row(income_row, ['epsdiluted', 'eps'])
            result['EPS'].append(eps)

            # Shares Outstanding (Diluted) - in millions
            shares = self._get_metric_from_row(income_row, ['weightedAverageShsOutDil', 'weightedAverageShsOut'])
            result['Shares Outstanding'].append(shares / 1_000_000 if shares else None)

            # Cash flow metrics
            if cashflow_row is not None:
                ocf = self._get_metric_from_row(cashflow_row, ['operatingCashFlow', 'netCashProvidedByOperatingActivities'])
                capex = self._get_metric_from_row(cashflow_row, ['capitalExpenditure', 'capitalExpenditures'])

                result['OCF'].append(ocf / 1_000_000 if ocf else None)
                result['CapEx'].append(capex / 1_000_000 if capex else None)

                if ocf is not None and capex is not None:
                    result['Free Cash Flow'].append((ocf - abs(capex)) / 1_000_000)
                else:
                    result['Free Cash Flow'].append(None)
            else:
                result['OCF'].append(None)
                result['CapEx'].append(None)
                result['Free Cash Flow'].append(None)

            # Balance sheet metrics
            if balance_row is not None:
                total_debt = self._get_metric_from_row(balance_row, ['totalDebt', 'longTermDebt'])
                cash = self._get_metric_from_row(balance_row, ['cashAndCashEquivalents', 'cash'])

                result['Gross Debt'].append(total_debt / 1_000_000 if total_debt else None)
                result['Cash'].append(cash / 1_000_000 if cash else None)

                if total_debt is not None and cash is not None:
                    net_debt = total_debt - cash
                    result['Net Debt'].append(net_debt / 1_000_000)
                else:
                    result['Net Debt'].append(None)
            else:
                result['Gross Debt'].append(None)
                result['Cash'].append(None)
                result['Net Debt'].append(None)

            # TTM calculations for ratios (need 4 quarters of history)
            # Only calculate for quarters where we have enough history
            if i >= 3:
                # Calculate TTM values
                ttm_revenue = sum([result['Sales'][j] for j in range(i - 3, i + 1)
                                   if result['Sales'][j] is not None])
                ttm_ebitda = sum([result['EBITDA'][j] for j in range(i - 3, i + 1)
                                  if result['EBITDA'][j] is not None])
                ttm_fcf = sum([result['Free Cash Flow'][j] for j in range(i - 3, i + 1)
                               if result['Free Cash Flow'][j] is not None])

                # Market Cap (use metrics if available, else quote)
                if metrics_row is not None and pd.notna(metrics_row.get('marketCap')):
                    market_cap = metrics_row['marketCap']
                else:
                    market_cap = quote.get('marketCap')

                result['Market Cap'].append(market_cap / 1_000_000 if market_cap else None)

                # EV
                net_debt_val = result['Net Debt'][-1]
                market_cap_val = result['Market Cap'][-1]

                if market_cap_val and net_debt_val is not None:
                    ev = market_cap_val + net_debt_val
                    result['EV'].append(ev)
                else:
                    result['EV'].append(None)

                # Net Leverage (Net Debt / TTM EBITDA)
                if net_debt_val is not None and ttm_ebitda and ttm_ebitda != 0:
                    result['Net Leverage'].append(net_debt_val / ttm_ebitda)
                else:
                    result['Net Leverage'].append(None)

                # EV/EBITDA
                ev_val = result['EV'][-1]
                if ev_val and ttm_ebitda and ttm_ebitda != 0:
                    result['EV/EBITDA'].append(ev_val / ttm_ebitda)
                else:
                    result['EV/EBITDA'].append(None)

                # P/S
                if market_cap_val and ttm_revenue and ttm_revenue != 0:
                    result['P/S'].append(market_cap_val / ttm_revenue)
                else:
                    result['P/S'].append(None)

                # FCF Yield
                if market_cap_val and market_cap_val != 0:
                    result['FCF Yield'].append((ttm_fcf / market_cap_val) * 100 if ttm_fcf else None)
                else:
                    result['FCF Yield'].append(None)

                # Dividend Yield
                if metrics_row is not None:
                    div_yield = self._get_metric_from_row(metrics_row, ['dividendYield'])
                    if div_yield is not None:
                        result['Dividend Yield'].append(div_yield * 100)
                    else:
                        result['Dividend Yield'].append(None)
                else:
                    result['Dividend Yield'].append(None)
            else:
                # First 3 quarters - not enough data for TTM ratios
                result['Market Cap'].append(None)
                result['EV'].append(None)
                result['Net Leverage'].append(None)
                result['EV/EBITDA'].append(None)
                result['P/S'].append(None)
                result['FCF Yield'].append(None)
                result['Dividend Yield'].append(None)

        return result
