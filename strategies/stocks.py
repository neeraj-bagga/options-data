from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class StockNameInfo:
    eq: str
    nfo: str
    eq_exchange: str
    fo_exchange: str
    lot_size: int

class StockName(Enum):
    NIFTY = "nifty"
    HDFC_BANK = "hdfc_bank"
    ICICI_BANK = "icici_bank"
    SBI = "sbi"
    AXIS_BANK = "axis_bank"
    KOTAK_BANK = "kotak_bank"
    INDUSIND_BANK = "indusind_bank"
    RELIANCE = "reliance"
    TCS = "tcs"
    INFOSYS = "infosys"
    HUL = "hul"
    LNT = "lnt"
    BAJAJ_FINANCE = "bajaj_finance"
    ADANI_ENTERPRISES = "adani_enterprises"
    MARUTI = "maruti"
    ASIAN_PAINTS = "asian_paints"
    BHARTI_AIRTEL = "bharti_airtel"
    BEL = "bel"
    CIPLA = "cipla"
    COAL_INDIA = "coal_india"
    DR_REDDY = "dr_reddy"
    GRASIM = "grasim"
    HERO_MOTOCORP = "hero_motocorp"
    IEX = "iex"
    IOC = "ioc"
    ITC = "itc"
    JSW_STEEL = "jsw_steel"
    LAURUS_LABS = "laurus_labs"
    LUPIN = "lupin"
    NMDC = "nmdc"
    NTPC = "ntpc"
    ONGC = "ongc"
    POWER_GRID = "power_grid"
    SHRIRAM_FINANCE = "shriram_finance"
    TATA_MOTORS = "tata_motors"
    TATA_STEEL = "tata_steel"
    TITAN = "titan"
    ULTRATECH = "ultratech"
    WIPRO = "wipro"
    YES_BANK = "yes_bank"
    ETERNAL = "eternal"
    VODAFONE_IDEA = "vodafone_idea"
    BANK_NIFTY = "bank_nifty"
    FIN_NIFTY = "fin_nifty"
    SENSEX = "sensex"

STOCK_NAME_MAP = {
    StockName.NIFTY: StockNameInfo(eq="NIFTY 50", nfo="NIFTY", eq_exchange="NSE", fo_exchange="NFO", lot_size=50),
    StockName.HDFC_BANK: StockNameInfo(eq="HDFCBANK", nfo="HDFCBANK", eq_exchange="NSE", fo_exchange="NFO", lot_size=550),
    StockName.ICICI_BANK: StockNameInfo(eq="ICICIBANK", nfo="ICICIBANK", eq_exchange="NSE", fo_exchange="NFO", lot_size=1375),
    StockName.SBI: StockNameInfo(eq="SBIN", nfo="SBIN", eq_exchange="NSE", fo_exchange="NFO", lot_size=1500),
    StockName.AXIS_BANK: StockNameInfo(eq="AXISBANK", nfo="AXISBANK", eq_exchange="NSE", fo_exchange="NFO", lot_size=725),
    StockName.KOTAK_BANK: StockNameInfo(eq="KOTAKBANK", nfo="KOTAKBANK", eq_exchange="NSE", fo_exchange="NFO", lot_size=400),
    StockName.INDUSIND_BANK: StockNameInfo(eq="INDUSINDBK", nfo="INDUSINDBK", eq_exchange="NSE", fo_exchange="NFO", lot_size=700),
    StockName.RELIANCE: StockNameInfo(eq="RELIANCE", nfo="RELIANCE", eq_exchange="NSE", fo_exchange="NFO", lot_size=250),
    StockName.TCS: StockNameInfo(eq="TCS", nfo="TCS", eq_exchange="NSE", fo_exchange="NFO", lot_size=150),
    StockName.INFOSYS: StockNameInfo(eq="INFY", nfo="INFY", eq_exchange="NSE", fo_exchange="NFO", lot_size=300),
    StockName.HUL: StockNameInfo(eq="HINDUNILVR", nfo="HINDUNILVR", eq_exchange="NSE", fo_exchange="NFO", lot_size=300),
    StockName.LNT: StockNameInfo(eq="LT", nfo="LT", eq_exchange="NSE", fo_exchange="NFO", lot_size=575),
    StockName.BAJAJ_FINANCE: StockNameInfo(eq="BAJFINANCE", nfo="BAJFINANCE", eq_exchange="NSE", fo_exchange="NFO", lot_size=125),
    StockName.ADANI_ENTERPRISES: StockNameInfo(eq="ADANIENT", nfo="ADANIENT", eq_exchange="NSE", fo_exchange="NFO", lot_size=250),
    StockName.MARUTI: StockNameInfo(eq="MARUTI", nfo="MARUTI", eq_exchange="NSE", fo_exchange="NFO", lot_size=100),
    StockName.ASIAN_PAINTS: StockNameInfo(eq="ASIANPAINT", nfo="ASIANPAINT", eq_exchange="NSE", fo_exchange="NFO", lot_size=350),
    StockName.BHARTI_AIRTEL: StockNameInfo(eq="BHARTIARTL", nfo="BHARTIARTL", eq_exchange="NSE", fo_exchange="NFO", lot_size=950),
    StockName.BEL: StockNameInfo(eq="BEL", nfo="BEL", eq_exchange="NSE", fo_exchange="NFO", lot_size=2850),
    StockName.CIPLA: StockNameInfo(eq="CIPLA", nfo="CIPLA", eq_exchange="NSE", fo_exchange="NFO", lot_size=650),
    StockName.COAL_INDIA: StockNameInfo(eq="COALINDIA", nfo="COALINDIA", eq_exchange="NSE", fo_exchange="NFO", lot_size=2100),
    StockName.DR_REDDY: StockNameInfo(eq="DRREDDY", nfo="DRREDDY", eq_exchange="NSE", fo_exchange="NFO", lot_size=125),
    StockName.GRASIM: StockNameInfo(eq="GRASIM", nfo="GRASIM", eq_exchange="NSE", fo_exchange="NFO", lot_size=475),
    StockName.HERO_MOTOCORP: StockNameInfo(eq="HEROMOTOCO", nfo="HEROMOTOCO", eq_exchange="NSE", fo_exchange="NFO", lot_size=300),
    StockName.IEX: StockNameInfo(eq="IEX", nfo="IEX", eq_exchange="NSE", fo_exchange="NFO", lot_size=2200),
    StockName.IOC: StockNameInfo(eq="IOC", nfo="IOC", eq_exchange="NSE", fo_exchange="NFO", lot_size=4375),
    StockName.ITC: StockNameInfo(eq="ITC", nfo="ITC", eq_exchange="NSE", fo_exchange="NFO", lot_size=1600),
    StockName.JSW_STEEL: StockNameInfo(eq="JSWSTEEL", nfo="JSWSTEEL", eq_exchange="NSE", fo_exchange="NFO", lot_size=850),
    StockName.LAURUS_LABS: StockNameInfo(eq="LAURUSLABS", nfo="LAURUSLABS", eq_exchange="NSE", fo_exchange="NFO", lot_size=950),
    StockName.LUPIN: StockNameInfo(eq="LUPIN", nfo="LUPIN", eq_exchange="NSE", fo_exchange="NFO", lot_size=425),
    StockName.NMDC: StockNameInfo(eq="NMDC", nfo="NMDC", eq_exchange="NSE", fo_exchange="NFO", lot_size=3000),
    StockName.NTPC: StockNameInfo(eq="NTPC", nfo="NTPC", eq_exchange="NSE", fo_exchange="NFO", lot_size=5700),
    StockName.ONGC: StockNameInfo(eq="ONGC", nfo="ONGC", eq_exchange="NSE", fo_exchange="NFO", lot_size=3850),
    StockName.POWER_GRID: StockNameInfo(eq="POWERGRID", nfo="POWERGRID", eq_exchange="NSE", fo_exchange="NFO", lot_size=3600),
    StockName.SHRIRAM_FINANCE: StockNameInfo(eq="SHRIRAMFIN", nfo="SHRIRAMFIN", eq_exchange="NSE", fo_exchange="NFO", lot_size=300),
    StockName.TATA_MOTORS: StockNameInfo(eq="TATAMOTORS", nfo="TATAMOTORS", eq_exchange="NSE", fo_exchange="NFO", lot_size=2850),
    StockName.TATA_STEEL: StockNameInfo(eq="TATASTEEL", nfo="TATASTEEL", eq_exchange="NSE", fo_exchange="NFO", lot_size=4250),
    StockName.TITAN: StockNameInfo(eq="TITAN", nfo="TITAN", eq_exchange="NSE", fo_exchange="NFO", lot_size=375),
    StockName.ULTRATECH: StockNameInfo(eq="ULTRACEMCO", nfo="ULTRACEMCO", eq_exchange="NSE", fo_exchange="NFO", lot_size=100),
    StockName.WIPRO: StockNameInfo(eq="WIPRO", nfo="WIPRO", eq_exchange="NSE", fo_exchange="NFO", lot_size=1500),
    StockName.YES_BANK: StockNameInfo(eq="YESBANK", nfo="YESBANK", eq_exchange="NSE", fo_exchange="NFO", lot_size=7800),
    StockName.ETERNAL: StockNameInfo(eq="ETERNAL", nfo="ETERNAL", eq_exchange="NSE", fo_exchange="NFO", lot_size=0),  # dummy, check NSE
    StockName.VODAFONE_IDEA: StockNameInfo(eq="IDEA", nfo="IDEA", eq_exchange="NSE", fo_exchange="NFO", lot_size=7000),
    StockName.BANK_NIFTY: StockNameInfo(eq="NIFTY BANK", nfo="BANKNIFTY", eq_exchange="NSE", fo_exchange="NFO", lot_size=15),
    StockName.FIN_NIFTY: StockNameInfo(eq="NIFTY FIN SERVICE", nfo="FINNIFTY", eq_exchange="NSE", fo_exchange="NFO", lot_size=40),
    StockName.SENSEX: StockNameInfo(eq="SENSEX", nfo="SENSEX", eq_exchange="BSE", fo_exchange="BFO", lot_size=10),
}

def get_lot_size(symbol: str) -> int:
    """Get lot size for a given symbol name."""
    # Create a mapping from symbol names to lot sizes
    symbol_to_lot = {
        'nifty': 50,
        'hdfc_bank': 550,
        'icici_bank': 1375,
        'sbi': 1500,
        'axis_bank': 725,
        'kotak_bank': 400,
        'indusind_bank': 700,
        'reliance': 250,
        'tcs': 150,
        'infosys': 300,
        'hul': 300,
        'lnt': 575,
        'bajaj_finance': 125,
        'adani_enterprises': 250,
        'maruti': 100,
        'asian_paints': 350,
        'bharti_airtel': 950,
        'bel': 2850,
        'cipla': 650,
        'coal_india': 2100,
        'dr_reddy': 125,
        'grasim': 475,
        'hero_motocorp': 300,
        'iex': 2200,
        'ioc': 4375,
        'itc': 1600,
        'jsw_steel': 850,
        'laurus_labs': 950,
        'lupin': 425,
        'nmdc': 3000,
        'ntpc': 5700,
        'ongc': 3850,
        'power_grid': 3600,
        'shriram_finance': 300,
        'tata_motors': 2850,
        'tata_steel': 4250,
        'titan': 375,
        'ultratech': 100,
        'wipro': 1500,
        'yes_bank': 7800,
        'eternal': 0,  # dummy
        'vodafone_idea': 7000,
        'bank_nifty': 15,
        'fin_nifty': 40,
        'sensex': 10,
    }
    return symbol_to_lot.get(symbol.lower(), 1)  # Default to 1 if not found
