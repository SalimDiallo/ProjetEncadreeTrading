"""Point d'entrée simple pour lancer le pipeline NLP."""
import os
import sys

# Ensure the directory containing the oil_sentiment_pipeline module is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from oil_sentiment_pipeline.main import main

if __name__ == "__main__":
    main()
