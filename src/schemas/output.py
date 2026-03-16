# src/schemas/output.py
# Pydantic model for the final rules bank output file.
#
# Fields (planned):
#   - rules: list[Rule]     — validated and structured rules
#   - metadata: dict        — source document info, generation date, model used
#   - statistics: dict      — extraction counts, validation error rate, iterations
#
# TODO: implement RulesBank output Pydantic model
