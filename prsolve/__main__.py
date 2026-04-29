import sys

if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--gui"):
    from .gui import launch
    launch()
else:
    from .cli import main
    main()
