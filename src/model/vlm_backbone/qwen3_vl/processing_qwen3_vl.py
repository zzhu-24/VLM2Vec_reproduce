# coding=utf-8
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
"""
Qwen3-VL processor for VLM2Vec.
This module provides a processor wrapper around transformers' native Qwen3-VL processor.
"""

from typing import List, Union, Optional, Dict, Any
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput
try:
    from transformers.video_utils import VideoInput
except ImportError:
    from transformers.image_utils import VideoInput

# Try to import Qwen3-VL processor from transformers
try:
    from transformers import Qwen3VLProcessor as TransformersQwen3VLProcessor
    HAS_NATIVE_QWEN3_PROCESSOR = True
except ImportError:
    HAS_NATIVE_QWEN3_PROCESSOR = False
    TransformersQwen3VLProcessor = None

# Fallback to AutoProcessor
from transformers import AutoProcessor


class Qwen3VLProcessor(ProcessorMixin):
    """
    Qwen3-VL processor wrapper for VLM2Vec.
    This class wraps transformers' native Qwen3-VL processor to ensure compatibility.
    """
    
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs
    ):
        """
        Initialize Qwen3-VL processor.
        
        Args:
            image_processor: Image processor instance
            tokenizer: Tokenizer instance
            chat_template: Chat template string
            **kwargs: Additional arguments
        """
        # Initialize using transformers native processor if available
        if HAS_NATIVE_QWEN3_PROCESSOR and TransformersQwen3VLProcessor is not None:
            if image_processor is None and tokenizer is None:
                # Will be loaded from model_name in from_pretrained
                self._processor = None
            else:
                self._processor = TransformersQwen3VLProcessor(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    **kwargs
                )
        else:
            # Fallback: create a simple wrapper
            self._processor = None
        
        # Set image and video tokens
        if tokenizer is not None:
            self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
            self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        else:
            self.image_token = "<|image_pad|>"
            self.video_token = "<|video_pad|>"
        
        # Initialize parent if processor is available
        if self._processor is not None:
            super().__init__(
                self._processor.image_processor,
                self._processor.tokenizer,
                chat_template=chat_template
            )
        else:
            # Will be initialized in from_pretrained
            super().__init__(image_processor, tokenizer, chat_template=chat_template)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        image_processor=None,
        tokenizer=None,
        size=None,
        **kwargs
    ):
        """
        Load processor from pretrained model.
        
        Args:
            model_name_or_path: Path to model or model identifier
            image_processor: Optional image processor instance
            tokenizer: Optional tokenizer instance
            size: Optional size configuration for image processor
            **kwargs: Additional arguments
        """
        # Try to use transformers native processor
        if HAS_NATIVE_QWEN3_PROCESSOR and TransformersQwen3VLProcessor is not None:
            try:
                processor = TransformersQwen3VLProcessor.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    **kwargs
                )
                # Handle size parameter for image processor if provided
                if size is not None and hasattr(processor.image_processor, 'size'):
                    # Update image processor size if it supports it
                    if hasattr(processor.image_processor, 'min_pixels') and 'shortest_edge' in size:
                        processor.image_processor.min_pixels = size.get('shortest_edge')
                    if hasattr(processor.image_processor, 'max_pixels') and 'longest_edge' in size:
                        processor.image_processor.max_pixels = size.get('longest_edge')
                    # Also set size dict if supported
                    if hasattr(processor.image_processor, 'size'):
                        # Only pass valid keys (shortest_edge, longest_edge)
                        valid_size = {k: v for k, v in size.items() if k in ['shortest_edge', 'longest_edge']}
                        if valid_size:
                            processor.image_processor.size = valid_size
                
                instance = cls.__new__(cls)
                instance._processor = processor
                instance.image_token = getattr(processor.tokenizer, "image_token", "<|image_pad|>")
                instance.video_token = getattr(processor.tokenizer, "video_token", "<|video_pad|>")
                # Initialize parent with processor components
                super(cls, instance).__init__(
                    processor.image_processor,
                    processor.tokenizer,
                    chat_template=getattr(processor, 'chat_template', None)
                )
                return instance
            except Exception:
                # Fallback to AutoProcessor
                pass
        
        # Fallback to AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs
        )
        
        instance = cls.__new__(cls)
        instance._processor = processor
        instance.image_processor = processor.image_processor
        instance.tokenizer = processor.tokenizer
        instance.image_token = getattr(processor.tokenizer, "image_token", "<|image_pad|>")
        instance.video_token = getattr(processor.tokenizer, "video_token", "<|video_pad|>")
        
        # Handle size parameter for image processor if provided
        if size is not None:
            # Only pass valid keys (shortest_edge, longest_edge) to avoid ValueError
            valid_size = {k: v for k, v in size.items() if k in ['shortest_edge', 'longest_edge']}
            if valid_size and hasattr(processor.image_processor, 'size'):
                processor.image_processor.size = valid_size
            # Also try to set min_pixels/max_pixels if the processor supports it
            if hasattr(processor.image_processor, 'min_pixels') and 'shortest_edge' in size:
                processor.image_processor.min_pixels = size.get('shortest_edge')
            if hasattr(processor.image_processor, 'max_pixels') and 'longest_edge' in size:
                processor.image_processor.max_pixels = size.get('longest_edge')
        
        return instance
    
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        return_tensors=None,
        **kwargs
    ):
        """Process inputs."""
        if self._processor is not None:
            return self._processor(
                images=images,
                text=text,
                videos=videos,
                return_tensors=return_tensors,
                **kwargs
            )
        else:
            # Fallback processing
            if images is not None:
                processed_images = self.image_processor(images, return_tensors=return_tensors, **kwargs)
            else:
                processed_images = {}
            
            if text is not None:
                processed_text = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            else:
                processed_text = {}
            
            # Merge results
            result = {**processed_images, **processed_text}
            return result
    
    def decode(self, *args, **kwargs):
        """Decode token ids to text."""
        if self._processor is not None:
            return self._processor.decode(*args, **kwargs)
        else:
            return self.tokenizer.decode(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        """Batch decode token ids to text."""
        if self._processor is not None:
            return self._processor.batch_decode(*args, **kwargs)
        else:
            return self.tokenizer.batch_decode(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying processor."""
        if name in ['_processor', 'image_processor', 'tokenizer', 'image_token', 'video_token']:
            return super().__getattribute__(name)
        try:
            return getattr(self._processor, name) if self._processor is not None else super().__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

