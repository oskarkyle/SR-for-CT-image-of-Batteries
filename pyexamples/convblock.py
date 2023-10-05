import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../sample_2.jpeg'), 
    to_Conv_init("init", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="init"),
    *conv_block(name='1', botton='init', s_filter=512, n_filter=64, offset="(1,0,0)", size=(40,40,6.5), opacity=0.5),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
