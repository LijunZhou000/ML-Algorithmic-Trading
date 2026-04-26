from tradepy.data.to_parquet import convert_all, convert_one
import logging
import argparse

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='Convertir txt a parquet')
    parser.add_argument('--convert-all', action='store_true',
                        help='Convertir todos los txt a parquet')
    parser.add_argument('--force', action='store_true',
                        help='Forzar reconversión aunque el parquet exista')
    parser.add_argument('--ticker', type=str, default=None,
                        help='Convertir un ticker concreto')
    args = parser.parse_args()

    if args.convert_all:
        convert_all(force=args.force)
    elif args.ticker:
        convert_one(args.ticker, force=args.force)
    else:
        parser.print_help()