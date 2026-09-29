"""Optional process-local OrcaSAQ2 ExLlamaV3 embedding loader patch."""
from orcasaq2.patches import int8_embedding
int8_embedding.apply()
