class DINOArgs:
    dino_resolution: int = 800
    dinov2_model_name: str = 'dinov2_vits14'
    upsampled: bool = False
    
    @classmethod
    def id_dict(cls):
        """Return dict that identifies the DINO model parameters."""
        return {
            "dino_resolution": cls.dino_resolution,
            "dinov2_model_name": cls.dinov2_model_name,
            "upsampled": cls.upsampled,
        }
