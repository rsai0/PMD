# %%
def get_Himawari_file(tt, Himawari_flag="H08", resolution=0.05, sign="cloud"):
    if sign == "RS":
        Himawari_file = "NC_"+Himawari_flag+tt.strftime("_%Y%m%d")
        Himawari_file = Himawari_file+tt.strftime("_%H%M")
        if resolution == 0.05:
            Himawari_file = Himawari_file+"_R21_FLDK.02401_02401.nc"
        elif resolution == 0.02:
            Himawari_file = Himawari_file+"_R21_FLDK.06001_06001.nc"
        else:
            raise ValueError("Resolution must be 0.02 or 0.05")
    elif sign == "cloud":
        Himawari_file = "NC_"+Himawari_flag+tt.strftime("_%Y%m%d")
        Himawari_file = Himawari_file+tt.strftime("_%H%M")
        if resolution == 0.05:
            Himawari_file = Himawari_file+"_L2CLP010_FLDK.02401_02401.nc"
        elif resolution == 0.02:
            Himawari_file = Himawari_file+"_L2CLP010_FLDK.06001_06001.nc"
        else:
            raise ValueError("Resolution must be 0.02 or 0.05")
    return Himawari_file



