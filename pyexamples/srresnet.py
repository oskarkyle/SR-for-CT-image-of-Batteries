import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input_color("input", s_filer=512, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption="input"), 
    to_Conv("init", s_filer=512, n_filer=64, offset="(1,0,0)", to="(input-east)", width=2, height=40, depth=40),
    to_ConvConvRelu("ccr_b1", s_filer=512, n_filer=(64, 64), offset="(1,0,0)", to="(init-east)", width=(2,2), height=40, depth=40, caption="conv1"),
    

    to_ConvConvRelu("ccr_b2", s_filer=512, n_filer=(64, 64), offset="(1,0,0)", to="(ccr_b1-east)", width=(2,2), height=40, depth=40, caption="conv2"),
    
    to_ConvConvRelu("ccr_b3", s_filer=512, n_filer=(64, 64), offset="(1,0,0)", to="(ccr_b2-east)", width=(2,2), height=40, depth=40, caption="conv3"),
   
    to_Sum("sum_b1", offset="(2,0,0)", to="(ccr_b3-east)", radius=5),
    to_Conv("fin", s_filer=512, n_filer=64, offset="(1.5,0,0)", to="(sum_b1-east)", width=2, height=40, depth=40),
    to_input_color("output", s_filer=512, n_filer=64, offset="(1,0,0)", to="(fin-east)", width=1, height=40, depth=40, caption="output"),

    to_connection("input", "init"),
    to_connection("init", "ccr_b1"),
    to_connection("ccr_b1", "ccr_b2"),
    to_connection("ccr_b2", "ccr_b3"),
    to_connection("ccr_b3", "sum_b1"),
    to_connection("sum_b1", "fin"),
    to_connection("fin", "output"),

    to_combine("input", "sum_b1", pos=1.25, h=2.25),


    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()