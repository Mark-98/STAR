"""STAR Stage 1 entrypoint; run from the repository root."""
import sys
from star.training.rl import main

if __name__ == '__main__':
    sys.argv[1:1] = ['--stage', 'stage1']
    main()
