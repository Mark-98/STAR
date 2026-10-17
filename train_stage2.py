"""STAR Stage 2 entrypoint; accepts the original action-distillation CLI."""
import sys
from star.training.rl import main

if __name__ == '__main__':
    sys.argv[1:1] = ['--stage', 'stage2']
    main()
