
from .tikzeng import *

#define new block
def block_2ConvPool( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
    to_ConvConvRelu( 
        name="ccr_{}".format( name ),
        s_filer=str(s_filer), 
        n_filer=(n_filer,n_filer), 
        offset=offset, 
        to="({}-east)".format( botton ), 
        width=(size[2],size[2]), 
        height=size[0], 
        depth=size[1],   
        ),    
    to_Pool(         
        name="{}".format( top ), 
        offset="(0,0,0)", 
        to="(ccr_{}-east)".format( name ),  
        width=1,         
        height=size[0] - int(size[0]/4), 
        depth=size[1] - int(size[0]/4), 
        opacity=opacity, ),
    to_connection( 
        "{}".format( botton ), 
        "ccr_{}".format( name )
        )
    ]


def block_Unconv( name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5 ):
    return [
        to_UnPool(  name='unpool_{}'.format(name),    offset=offset,    to="({}-east)".format(botton),         width=1,              height=size[0],       depth=size[1], opacity=opacity ),
        to_ConvRes( name='ccr_res_{}'.format(name),   offset="(0,0,0)", to="(unpool_{}-east)".format(name),    s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='ccr_{}'.format(name),       offset="(0,0,0)", to="(ccr_res_{}-east)".format(name),   s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_ConvRes( name='ccr_res_c_{}'.format(name), offset="(0,0,0)", to="(ccr_{}-east)".format(name),       s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),       
        to_Conv(    name='{}'.format(top),            offset="(0,0,0)", to="(ccr_res_c_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection( 
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
            )
    ]




def block_Res( num, name, botton, top, s_filer=256, n_filer=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5 ):
    lys = []
    layers = [ *[ '{}_{}'.format(name,i) for i in range(num-1) ], top]
    for name in layers:        
        ly = [ to_Conv( 
            name='{}'.format(name),       
            offset=offset, 
            to="({}-east)".format( botton ),   
            s_filer=str(s_filer), 
            n_filer=str(n_filer), 
            width=size[2],
            height=size[0],
            depth=size[1]
            ),
            to_connection( 
                "{}".format( botton  ), 
                "{}".format( name ) 
                )
            ]
        botton = name
        lys+=ly
    
    lys += [
        to_skip( of=layers[1], to=layers[-2], pos=1.25),
    ]
    return lys

def Block_decoder(name, botton, top, s_filer=256, n_filer=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5):
    return[
        to_UnPool(name='unpool_{}'.format(name), offset=offset, to="({}-east)".format(botton), width=1, height=size[0], depth=size[1], opacity=opacity),
        to_ConvRes( name='ccr_res_{}'.format(name), offset="(0,0,0)", to="(unpool_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1], opacity=opacity ),
        to_Conv( name='ccr_{}'.format(name), offset="(0,0,0)", to="(ccr_res_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_Conv( name='{}'.format(top), offset="(0,0,0)", to="(ccr_{}-east)".format(name), s_filer=str(s_filer), n_filer=str(n_filer), width=size[2], height=size[0], depth=size[1] ),
        to_connection(
            "{}".format( botton ), 
            "unpool_{}".format( name ) 
        )
    ]

def block_skip_connection(name, top, botton, s_filter=256, n_filter=64, offset="(1,0,0)", size=(32,32,3.5), opacity=0.5):
    return [
        to_GNRelu(name='gnrelu_{}'.format(name), 
                  s_filer=str(s_filter),
                  n_filer=str(n_filter),
                  offset=offset,
                  to="{}-east".format(botton),
                  width=size[2],
                  height=size[0],
                  depth=size[1],
                  ),
        to_Pool(name="pool_{}".format(top), 
                offset="(0,0,0)", 
                to="gnrelu_{}-east".format(name), 
                width=1, )

    ]

def conv_block(name, botton, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5):
    return [
        to_Conv("conv1_{}".format(name), 
                s_filer=str(s_filter), 
                n_filer=str(n_filter), 
                offset=offset, 
                to="({}-east)".format(botton), 
                height=size[0], 
                depth=size[1], 
                width=size[2],
                caption="CB"),

        to_GNRelu_in_block("gn_{}".format(name), 
                  s_filer=str(s_filter), 
                  n_filer=str(n_filter), 
                  offset="(0,0,0)", 
                  to="(conv1_{}-east)".format(name), 
                  height=size[0], 
                  depth=size[1], 
                  width=1.5 ),

        to_Pool("pool_{}".format(name), 
                offset="(0,0,0)",
                to="(gn_{}-east)".format(name), 
                height=size[0] , 
                depth=size[1], 
                width=1),

        to_connection(botton, "conv1_{}".format(name))
    ]

def skip_connection_block(name, botton, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5):
    return [
        to_GNRelu(  "gn_{}".format(name), 
                    s_filer=str(s_filter),
                    n_filer=str(n_filter),
                    offset=offset,
                    to="(pool_{}-east)".format(botton),
                    height=size[0],
                    depth=size[1],
                    width=size[2],
                     caption="Skip" ),

        to_Pool(    "pool_{}".format(name),
                    offset="(0,0,0)",
                    to="(gn_{}-east)".format(name),
                    height=size[0] - int(size[0]/2),
                    depth=size[0] - int(size[0]/2),
                    width=1,
                    opacity=opacity)

    ]

def combine_connection_block(name, botton, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5):
    return [
        to_UnPool("unpool_{}".format(name), 
                  offset=offset, 
                  to="(pool_{}-east)".format(botton), 
                  height=size[0], 
                  depth=size[1], 
                  width=1,
                  caption="Combine"),


        to_GNRelu("gn_{}".format(name),
                   s_filer=str(s_filter), 
                   n_filer=str(n_filter), 
                   offset="(0,0,0)", 
                   to="(unpool_{}-east)".format(name), 
                   height=size[0], 
                   depth=size[1], 
                   width=size[2]),


        to_Sum(
            "sum_{}".format(name),
            offset="(2,0,0)",
            to="(gn_{}-east)".format(name),
            radius=5,
            opacity=0.6
        ),

        to_connection("gn_{}".format(name), "sum_{}".format(name))
    ]

def to_middle_block(name, botton, s_filter=256, n_filter=64, offset="(0,0,0)", size=(32,32,3.5), opacity=0.5):
    return [


    ]