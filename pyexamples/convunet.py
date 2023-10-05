import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    to_Conv("init", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=80, depth=80, width=3.5, caption="init"),
    #layer 1
    *conv_block(name='b1', botton='init', s_filter=512, n_filter=64, offset="(1,0,0)", size=(80,80,3.5), opacity=0.5),
    *skip_connection_block(name='scb1', botton='b1', s_filter=512, n_filter=64, offset="(0,0,0)", size=(80,80,3.5), opacity=0.5),

    #layer 2
    *conv_block(name='b2', botton='pool_scb1', s_filter=256, n_filter=128, offset="(1.1,0,0)", size=(40,40,5.5), opacity=0.5),
    *skip_connection_block(name='scb2', botton='b2', s_filter=256, n_filter=128, offset="(0,0,0)", size=(40,40,5.5), opacity=0.5),

    #layer 3
    *conv_block(name='b3', botton='pool_scb2', s_filter=128, n_filter=256, offset="(1.2,0,0)", size=(20,20,7.5), opacity=0.5),
    *skip_connection_block(name='scb3', botton='b3', s_filter=128, n_filter=256, offset="(0,0,0)", size=(20,20,7.5), opacity=0.5),

    #layer 4
    *conv_block(name='b4', botton='pool_scb3', s_filter=64, n_filter=512, offset="(1.3,0,0)", size=(10,10,9.5), opacity=0.5),
    *skip_connection_block(name='scb4', botton='b4', s_filter=64, n_filter=512, offset="(0,0,0)", size=(10,10,9.5), opacity=0.5),

    #middle
    to_Conv(name='middle',s_filer=32,n_filer=1024,offset="(1.4,0,0)",to="(pool_scb4-east)",height=5,depth=5,width=11.5,caption="middle"),
    to_GNRelu_in_block(name='gn_middle',s_filer=32,n_filer=1024,offset="(0,0,0)",to="(middle-east)",height=5,depth=5,width=1.5),
    to_Pool(name="pool_middle", offset="(0,0,0)", to="(gn_middle-east)", width=1, height=5, depth=5, opacity=0.5),
    to_connection('pool_scb4','middle'),

    #layer 4
    *conv_block(name='cb4', botton='pool_middle', s_filter=32, n_filter=512, offset="(1.45,0,0)", size=(5,5,9.5), opacity=0.5),
    *combine_connection_block(name='ccb4', botton='cb4', s_filter=64, n_filter=512, offset="(0,0,0)", size=(10,10,9.5), opacity=0.5),

    #layer 3
    *conv_block(name='cb3', botton='sum_ccb4', s_filter=64, n_filter=256, offset="(1.5,0,0)", size=(10,10,7.5), opacity=0.5),
    *combine_connection_block(name='ccb3', botton='cb3', s_filter=128, n_filter=256, offset="(0,0,0)", size=(20,20,7.5), opacity=0.5),

    #layer 2
    *conv_block(name='cb2', botton='sum_ccb3', s_filter=128, n_filter=128, offset="(1.6,0,0)", size=(20,20,5.5), opacity=0.5),
    *combine_connection_block(name='ccb2', botton='cb2', s_filter=256, n_filter=128, offset="(0,0,0)", size=(40,40,5.5), opacity=0.5),

    #layer 1
    *conv_block(name='cb1', botton='sum_ccb2', s_filter=256, n_filter=64, offset="(1.65,0,0)", size=(40,40,3.5), opacity=0.5),
    *combine_connection_block(name='ccb1', botton='cb1', s_filter=512, n_filter=64, offset="(0,0,0)", size=(80,80,3.5), opacity=0.5),

    to_Conv(name='fin_conv', s_filer=512, n_filer=64, offset="(1.7,0,0)", to="(sum_ccb1-east)", height=80, depth=80, width=3.5),
    to_GNRelu_in_block(name='fin_gn',s_filer=512, n_filer=64, offset="(0,0,0)",to="(fin_conv-east)", height=80, depth=80, width=1.5, caption="final"),
    
    to_connection('sum_ccb1', 'fin_conv'),

    to_combine("gn_scb4", "sum_ccb4", pos=1.25, h=0),
    to_combine("gn_scb3", "sum_ccb3", pos=1.25, h=0.75),
    to_combine("gn_scb2", "sum_ccb2", pos=1.25, h=2.25),
    to_combine("gn_scb1", "sum_ccb1", pos=1.25, h=5.25),
    to_end()

]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()