
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input( '../results/sample.jpeg'), 
    to_Conv_init("init", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="init"),
    #conv block 1
    *conv_block(name='b1', botton='init', s_filter=512, n_filter=64, offset="(1,0,0)", size=(40,40,3.5), opacity=0.5),
    

    #skip connection block
    *skip_connection_block(name='scb1', botton='b1', s_filter=512, n_filter=64, offset="(2,0,0)", size=(40,40,3.5), opacity=0.5),
    

    #middle conv block
    to_Conv(name='middle', s_filer=256, n_filer=128, offset="(2,0,0)", to="(pool_scb1-east)", height=20, depth=20, width=6.5, caption="middle"),
    to_GNRelu_in_block(name='gn_middle',s_filer=256, n_filer=128, offset="(0,0,0)",to="(middle-east)", height=20, depth=20, width=1.5, caption=""),
    to_Pool(name="pool_middle", offset="(0,0,0)", to="(gn_middle-east)", width=1, height=20, depth=20, opacity=0.5, caption=" "),

    #conv block 2
    *conv_block(name='b2', botton='pool_middle', s_filter=256, n_filter=64, offset="(1,0,0)", size=(20,20,3.5), opacity=0.5),

    #combine connection block
    *combine_connection_block(name='ccb1', botton='b2', s_filter=512, n_filter=64, offset="(2,0,0)", size=(40,40,3.5), opacity=0.5),

    #final
    to_Conv_init("final", 512, 64, offset="(1,0,0)", to="(sum_ccb1-east)", height=40, depth=40, width=2, caption="final"),

    to_connection("pool_b1", "gn_scb1"),
    to_connection("pool_scb1", "middle"),
    to_connection("pool_b2", "gn_ccb1"),
    to_connection("sum_ccb1", "final"),
    
    #to_skip("gn2", "combine1", pos=1.25),
    to_combine("gn_scb1", "sum_ccb1", pos=1.25, h=2.25),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
