from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class StockNameInfo:
    eq: str
    nfo: str
    eq_exchange: str
    fo_exchange: str

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
    StockName.NIFTY: StockNameInfo(eq="NIFTY 50", nfo="NIFTY", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.HDFC_BANK: StockNameInfo(eq="HDFCBANK", nfo="HDFCBANK", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ICICI_BANK: StockNameInfo(eq="ICICIBANK", nfo="ICICIBANK", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.SBI: StockNameInfo(eq="SBIN", nfo="SBIN", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.AXIS_BANK: StockNameInfo(eq="AXISBANK", nfo="AXISBANK", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.KOTAK_BANK: StockNameInfo(eq="KOTAKBANK", nfo="KOTAKBANK", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.INDUSIND_BANK: StockNameInfo(eq="INDUSINDBK", nfo="INDUSINDBK", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.RELIANCE: StockNameInfo(eq="RELIANCE", nfo="RELIANCE", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.TCS: StockNameInfo(eq="TCS", nfo="TCS", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.INFOSYS: StockNameInfo(eq="INFY", nfo="INFY", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.HUL: StockNameInfo(eq="HINDUNILVR", nfo="HINDUNILVR", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.LNT: StockNameInfo(eq="LT", nfo="LT", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.BAJAJ_FINANCE: StockNameInfo(eq="BAJFINANCE", nfo="BAJFINANCE", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ADANI_ENTERPRISES: StockNameInfo(eq="ADANIENT", nfo="ADANIENT", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.MARUTI: StockNameInfo(eq="MARUTI", nfo="MARUTI", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ASIAN_PAINTS: StockNameInfo(eq="ASIANPAINT", nfo="ASIANPAINT", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.BHARTI_AIRTEL: StockNameInfo(eq="BHARTIARTL", nfo="BHARTIARTL", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.BEL: StockNameInfo(eq="BEL", nfo="BEL", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.CIPLA: StockNameInfo(eq="CIPLA", nfo="CIPLA", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.COAL_INDIA: StockNameInfo(eq="COALINDIA", nfo="COALINDIA", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.DR_REDDY: StockNameInfo(eq="DRREDDY", nfo="DRREDDY", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.GRASIM: StockNameInfo(eq="GRASIM", nfo="GRASIM", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.HERO_MOTOCORP: StockNameInfo(eq="HEROMOTOCO", nfo="HEROMOTOCO", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.IEX: StockNameInfo(eq="IEX", nfo="IEX", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.IOC: StockNameInfo(eq="IOC", nfo="IOC", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ITC: StockNameInfo(eq="ITC", nfo="ITC", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.JSW_STEEL: StockNameInfo(eq="JSWSTEEL", nfo="JSWSTEEL", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.LAURUS_LABS: StockNameInfo(eq="LAURUSLABS", nfo="LAURUSLABS", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.LUPIN: StockNameInfo(eq="LUPIN", nfo="LUPIN", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.NMDC: StockNameInfo(eq="NMDC", nfo="NMDC", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.NTPC: StockNameInfo(eq="NTPC", nfo="NTPC", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ONGC: StockNameInfo(eq="ONGC", nfo="ONGC", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.POWER_GRID: StockNameInfo(eq="POWERGRID", nfo="POWERGRID", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.SHRIRAM_FINANCE: StockNameInfo(eq="SHRIRAMFIN", nfo="SHRIRAMFIN", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.TATA_MOTORS: StockNameInfo(eq="TATAMOTORS", nfo="TATAMOTORS", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.TATA_STEEL: StockNameInfo(eq="TATASTEEL", nfo="TATASTEEL", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.TITAN: StockNameInfo(eq="TITAN", nfo="TITAN", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ULTRATECH: StockNameInfo(eq="ULTRACEMCO", nfo="ULTRACEMCO", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.WIPRO: StockNameInfo(eq="WIPRO", nfo="WIPRO", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.YES_BANK: StockNameInfo(eq="YESBANK", nfo="YESBANK", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.ETERNAL: StockNameInfo(eq="ETERNAL", nfo="ETERNAL", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.VODAFONE_IDEA: StockNameInfo(eq="IDEA", nfo="IDEA", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.BANK_NIFTY: StockNameInfo(eq="NIFTY BANK", nfo="BANKNIFTY", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.FIN_NIFTY: StockNameInfo(eq="NIFTY FIN SERVICE", nfo="FINNIFTY", eq_exchange="NSE", fo_exchange="NFO"),
    StockName.SENSEX: StockNameInfo(eq="SENSEX", nfo="SENSEX", eq_exchange="BSE", fo_exchange="BFO"),
}
