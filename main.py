import os
import argparse
import pprint

from model import get_encoded_data, train_model, get_model


def parse_args(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name)
    parser.add_argument('--domain', type=str, default='electronics', help='domain')
    parser.add_argument('--clex', type=float, default=1., help='lex loss coef')
    parser.add_argument('--cdoc', type=float, default=1., help='doc loss coef')
    parser.add_argument('--crel', type=float, default=1., help='rel loss coef')

    parser.add_argument('--embed_dim', type=int, default=300, help='embedding dimension')
    parser.add_argument('--rnn-dim', type=int, default=300, help='rnn output dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--rnn-dropout', type=float, default=0.2, help='rnn dropout rate')

    parser.add_argument('--reduce-patience', type=int, default=3, help='max epochs to wait before reduce lr')
    parser.add_argument('--stop-patience', type=int, default=5, help='max epochs to wait before stop training')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch-size')
    parser.add_argument('--max-epochs', type=int, default=100, help='maximal training epochs')


    args = parser.parse_args()
    pprint.pprint(args)

    # avoid domain input errors
    if args.domain in 'books':
        args.domain = 'books'
    elif args.domain in 'dvds':
        args.domain = 'dvds'
    elif args.domain in 'electronics':
        args.domain = 'electronics'
    else:
        raise RuntimeError('domain must one of books/dvds/electronics')
    return args


def main(args):
    ddata = get_encoded_data(args)

    train_model(
        args=args,
        ddata=ddata,
        gen_model_func=get_model,
    )


if __name__ == '__main__':
    _args = parse_args('Sent')
    main(_args)
