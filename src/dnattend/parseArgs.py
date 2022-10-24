#!/usr/bin/env python3

""" DNAttend """

import sys
import logging
import argparse
from timeit import default_timer as timer
from .main import train_cli, test_cli, retrain_cli, predict_cli, simulate_cli
from ._version import __version__


def parseArgs() -> argparse.Namespace:
    epilog = 'Stephen Richer, NHS England (stephen.richer@nhs.net)'
    baseParser = getBaseParser(__version__)
    parser = argparse.ArgumentParser(
        epilog=epilog, description=__doc__, parents=[baseParser])
    subparser = parser.add_subparsers(
        title='required commands',
        description='',
        dest='command',
        metavar='Commands',
        help='Description:')

    sp1 = subparser.add_parser(
        'train',
        description=train_cli.__doc__,
        help='Train model.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp1.add_argument(
        'config', help='YAML configuration file.')
    sp1.set_defaults(function=train_cli)


    sp2 = subparser.add_parser(
        'test',
        description=test_cli.__doc__,
        help='Test model.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp2.add_argument(
        'config', help='YAML configuration file.')
    sp2.set_defaults(function=test_cli)


    sp3 = subparser.add_parser(
        'retrain',
        description=retrain_cli.__doc__,
        help='Retrain model.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp3.add_argument(
        'config', help='YAML configuration file.')
    sp3.set_defaults(function=retrain_cli)


    sp4 = subparser.add_parser(
        'predict',
        description=predict_cli.__doc__,
        help='Run predictions model.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp4.add_argument(
        'data', help='Data to generate predictions.')
    sp4.add_argument(
        'model', help='Trained model in pickl format.')
    sp4.add_argument(
        '--verify', action='store_true',
        help='Validat output against example hash (default: %(default)s)')
    sp4.add_argument(
        '--sep', default=',',
        help='Seperator of input dats (default: %(default)s)')
    sp4.set_defaults(function=predict_cli)


    sp5 = subparser.add_parser(
        'simulate',
        description=simulate_cli.__doc__,
        help='Simulate test data.',
        parents=[baseParser],
        epilog=parser.epilog)
    sp5.add_argument(
        '--config',
        help='Path to write default config file (default: stderr)')
    sp5.add_argument(
        '--size', type=int, default=50_000,
        help='Number of records to simulate (default: %(default)s)')
    sp5.add_argument(
        '--noise', type=float, default=0.2,
        help='Scale factor for random noise (default: %(default)s)')
    sp5.add_argument(
        '--seed', type=int, default=42,
        help='Seed for random number generator (default: %(default)s)')
    sp5.set_defaults(function=simulate_cli)

    args = parser.parse_args()
    if 'function' not in args:
        parser.print_help()
        sys.exit()

    rc = executeCommand(args)
    return rc


def executeCommand(args):
    # Initialise logging
    logFormat = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    logging.basicConfig(level=args.verbose, format=logFormat)
    del args.verbose, args.command
    # Pop main function and excute script
    function = args.__dict__.pop('function')
    start = timer()
    rc = function(**vars(args))
    end = timer()
    logging.info(f'Total execution time: {end - start:.3f} seconds.')
    logging.shutdown()
    return rc


def getBaseParser(version: str) -> argparse.Namespace:
    """ Create base parser of verbose/version. """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--version', action='version', version='%(prog)s {}'.format(version))
    parser.add_argument(
        '--verbose', action='store_const', const=logging.INFO,
        default=logging.ERROR, help='verbose logging for debugging')
    return parser
