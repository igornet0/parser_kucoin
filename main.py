import asyncio
from sys import argv
from argparse import ArgumentParser

from handlers.att_parser import AttParser
from handlers.parser_handler import Handler as HandlerParser
from core.utils.configure_logging import setup_logging

# from core import DataManager

def print_help():
    print("""Usage: python start_parser.py <time> [--pause=int] [--count=int] [--mode=api|driver]
            Arguments:
            time        : Time parameter for the parser (default: 5m)
            Options:
            --pause     : Pause interval in minute (default: 60)
            --save      : Save dataset (default: 0)
            --save_type : Path to save dataset (default: raw)
            --count     : Count parser datasets for coin, -1 - infinity (default: 100)
            --mode      : Parser mode - 'api' or 'driver' (default: 'api')
            --clear     : Clear dataset for all coins (default: 0)
            --help      : Show this help message"""
          )


async def start_parser(count: int, time_parser="5m", pause=60, mode="api",
                       clear: bool = False, save: bool = 0, save_type: str = "raw"):

    if not mode in ["api", "driver"]:
        raise ValueError(f"Unknown mode: {mode}")
    
    parser = HandlerParser.get_parser(f"parser kucoin {mode}")

    att = AttParser(parser, pause, clear)

    data = await att.parse(count=count, time_parser=time_parser, 
                            save=save, save_type=save_type)
    
    if not data:
        return
    
    for coin, dataset in data.items():
        print(f"[INFO] {coin=}, {len(dataset)=}, type={type(dataset)}")


if __name__ == "__main__":    
    parser = ArgumentParser(description='Coin Parser', add_help=False)
    parser.add_argument('time', nargs='?', default="5m", help='Time parameter (default: 5m)')
    parser.add_argument('--pause', type=int, default=60, help='Pause interval (default: 60)')
    parser.add_argument('--count', type=int, default=100, help='Count parser datasets for coin, -1 - infinity (default: 100)')
    parser.add_argument('--save', type=int, default=0, help='Save dataset (default: 0)')
    parser.add_argument('--save_type', default='raw', help='Path to save dataset (default: raw)')
    parser.add_argument('--mode', default='api', choices=['api', 'driver'], help="Parser mode (default: 'api')")
    parser.add_argument('--clear',type=int, default=0, help='Clear dataset for all coins (default: 0)')
    
    if "--help" in argv or "-h" in argv:
        print_help()
        exit(0)

    setup_logging()
    
    # Парсинг аргументов
    args = parser.parse_args()
    asyncio.run(start_parser(args.count, args.time, args.pause, mode=args.mode,
                             clear=args.clear, save=args.save, save_type=args.save_type))
