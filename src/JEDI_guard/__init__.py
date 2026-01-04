# -*- coding: utf-8 -*-
"""
该包实现了 JEDI (Jailbreak dEfense via Detection and Intervention) 防御机制。

此 __init__.py 文件使 'Guard' 类
可以从包的顶层直接导入，方便用户使用。
"""

from .guard import Guard

__all__ = ["Guard"]
