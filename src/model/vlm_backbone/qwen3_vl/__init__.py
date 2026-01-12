# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

from src.model.vlm_backbone.qwen3_vl.modeling_qwen3_vl import *
from src.model.vlm_backbone.qwen3_vl.processing_qwen3_vl import *

if TYPE_CHECKING:
    from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from .processing_qwen3_vl import Qwen3VLProcessor

