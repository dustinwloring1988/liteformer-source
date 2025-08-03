# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ..utils import _LazyModule
from ..utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .auto import *
    from .bark import *
    from .bart import *
    from .chameleon import *
    from .clip import *
    from .clipseg import *
    from .gemma import *
    from .gemma2 import *
    from .gemma3 import *
    from .gemma3n import *
    from .llama import *
    from .llama4 import *
    from .llava import *
    from .llava_next import *
    from .llava_next_video import *
    from .llava_onevision import *
    from .mistral import *
    from .mistral3 import *
    from .mixtral import *
    from .mllama import *
    from .mobilenet_v1 import *
    from .mobilenet_v2 import *
    from .mobilevit import *
    from .mobilevitv2 import *
    from .paligemma import *
    from .phi import *
    from .phi3 import *
    from .phi4_multimodal import *
    from .phimoe import *
    from .pix2struct import *
    from .pixtral import *
    from .qwen2 import *
    from .qwen2_5_omni import *
    from .qwen2_5_vl import *
    from .qwen2_audio import *
    from .qwen2_moe import *
    from .qwen2_vl import *
    from .qwen3 import *
    from .qwen3_moe import *
    from .sam import *
    from .sam_hq import *
    from .shieldgemma2 import *
    from .siglip import *
    from .siglip2 import *
    from .smolvlm import *
    from .timm_backbone import *
    from .timm_wrapper import *
    from .video_llava import *
    from .vit import *
    from .vit_mae import *
    from .vit_msn import *
    from .vitdet import *
    from .vitmatte import *
    from .vitpose import *
    from .vitpose_backbone import *
    from .vits import *
    from .vivit import *
    from .whisper import *
    from .yolos import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
from . import aformer
from . import bformer
