import logging
import torch
from FlagEmbedding import BGEM3FlagModel
import os

# Patch os.symlink for Windows to bypass WinError 1314 when Developer Mode is off
if os.name == 'nt':
    _original_symlink = os.symlink
    def _patched_symlink(src, dst, target_is_directory=False, **kwargs):
        try:
            _original_symlink(src, dst, target_is_directory=target_is_directory, **kwargs)
        except OSError as e:
            if getattr(e, 'winerror', None) == 1314:
                import shutil
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                raise e
    os.symlink = _patched_symlink

logger = logging.getLogger(__name__)


# global model manager
class ModelManager:
    _embed_model = None

    @classmethod
    def get_embed_model(cls):
        """
        singleton pattern to load the embedding model only once.
        """
        if cls._embed_model is None:
            logger.info("loading BGE-M3 embedding model for the first time...")
            use_fp16 = torch.cuda.is_available()
            cls._embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16)
            logger.info("BGE-M3 embedding model loaded successfully!")
        return cls._embed_model
