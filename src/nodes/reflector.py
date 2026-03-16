# src/nodes/reflector.py
# Reflector Node: Applies self-reflection and Chain-of-Thought reasoning over
# extracted rules before they reach the Validator Node.
#
# Responsibilities:
#   - For each raw rule, retrieve relevant context from Qdrant (swagger.io + 3GPP)
#   - Use CoT reasoning to assess rule consistency and completeness
#   - Flag uncertain rules for priority validation
#   - Output reflected_rules list with confidence scores and reasoning traces
#
# TODO: implement reflector node logic
