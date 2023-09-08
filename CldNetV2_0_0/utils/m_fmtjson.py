# %%
"""
Created on Wed June 15 15:39:20 2022
格式化json格式, 便于查看
@author: BEOH
"""
import json5


def m_fmtjson(jsondata):
    """m_fmtjson
    格式化输出字典
    """
    return json5.dumps(
        jsondata,
        ensure_ascii=False,
        sort_keys=False,
        indent=2,
        separators=(",", ":"),
        trailing_commas=False,
    )



