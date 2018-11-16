#include <iostream>
#include <cstdlib>
#include <sstream>
#include <sys/time.h>
#include <mkldnn.hpp>
#include<iostream>
using namespace std;
using namespace mkldnn;

const int args_count = 5;
const int warmups = 20;
const int iters = 300;

void run(int seq_length_max , int batch, int input_size, int feature_size){
	// hidden layer feature's size =input feature size
	
	// RNN primitive characteristics
	int layers = 1;
	int direc = 1;
	int gates = 4;
	int num_states =2;
	
	auto cpu_engine = engine(engine::cpu, 0);
    auto null_memory_ = null_memory(cpu_engine);
	
	bool is_training = true;
    auto fwd_inf_train = is_training
                         ? prop_kind::forward_training
                         : prop_kind::forward_inference;
						 
	std::vector<primitive> fwd_net;
    std::vector<primitive> bwd_net;

	// mkldnn_tnc,(seq_length, batch, input channels)
	memory::dims net_src_dims = {seq_length_max, batch, input_size};
	//mkldnn_ldigo, (num_layers, num_directions,input_chanels, num_gates, output_channels)
	memory::dims common_weights_iter_dims = {layers, direc, feature_size, gates, feature_size};
	memory::dims common_weights_layer_dims = {layers, direc, input_size, gates, feature_size};
	//mkldnn_ldgo,  (num_layers, num_directions, num_gates, output_channels).
	memory::dims common_bias_dims = {layers, direc, gates, feature_size};
	
	// mkldnn_tnc,(seq_length, batch, input channels)
	memory::dims dst_layer_dims = {seq_length_max, batch, direc*feature_size};
	
	// mkldnn_ldsnc,(num_layers, num_directions, num_states,batch, state channels).
	memory::dims src_iter_dims = {layers, direc, num_states, batch, feature_size};
	memory::dims dst_iter_dims = {layers, direc, num_states, batch, feature_size};
	
	
	// multiplication of tensor dimensions
    auto tz_volume = [=](memory::dims tz_dims) {
        return std::accumulate(
            tz_dims.begin(), tz_dims.end(),
            (size_t)1, std::multiplies<size_t>());
    };

    // Create auxillary f32 memory descriptor
    // based on user- supplied dimensions and layout.
    auto formatted_md = [=](memory::dims dimensions, memory::format layout) {
        return memory::desc({ dimensions }, memory::data_type::f32, layout);
    };
    // Create auxillary generic f32 memory descriptor
    // based on supplied dimensions, with format::any.
    auto generic_md = [=](memory::dims dimensions) {
        return formatted_md( dimensions, memory::format::any);
    };
	
	// Net input
    std::vector<float> net_src(
            tz_volume(net_src_dims),
            1.0f);
	auto net_src_memory
        = mkldnn::memory({ formatted_md(net_src_dims, memory::format::tnc),
                           cpu_engine }, net_src.data());
						   
	// Other user provided memory arrays, desrciptors and primitives with the
    // data layouts chosen by user. We'll have to reorder if RNN
    // primitive prefers it in a different format.
	std::vector<float> user_common_weights_layer(
            tz_volume(common_weights_layer_dims),
            1.0f);
    auto user_common_weights_layer_memory
        = mkldnn::memory({ formatted_md(common_weights_layer_dims,
                           memory::format::ldigo), cpu_engine },
                         user_common_weights_layer.data());
    
	std::vector<float> user_common_weights_iter(
            tz_volume(common_weights_iter_dims),
            1.0f);
    auto user_common_weights_iter_memory
        = mkldnn::memory({ formatted_md(common_weights_iter_dims,
                           memory::format::ldigo), cpu_engine },
                         user_common_weights_iter.data());
	
    std::vector<float> user_common_bias(
            tz_volume(common_bias_dims),
            1.0f);
    auto user_common_bias_memory
        = mkldnn::memory({ formatted_md(common_bias_dims, memory::format::ldgo),
                           cpu_engine }, user_common_bias.data());
 
    std::vector<float> user_dst_layer(
            tz_volume(dst_layer_dims),
            1.0f);
    auto user_dst_layer_memory
        = mkldnn::memory({
                    formatted_md(dst_layer_dims, memory::format::tnc),
                    cpu_engine }, user_dst_layer.data());

	/* create a RNN forward primitive descriptor */			
	rnn_cell::desc uni_cell(algorithm::vanilla_lstm);

	rnn_forward::desc layer_desc(
        /* aprop_kind         */ fwd_inf_train,
        /* cell               */ uni_cell,
        /* direction          */ rnn_direction::unidirectional_left2right,
        /* src_layer_desc     */ formatted_md(net_src_dims, memory::format::tnc),
        /* src_iter_desc      */ generic_md(src_iter_dims),
        /* weights_layer_desc */ generic_md(common_weights_layer_dims),
        /* weights_iter_desc  */ generic_md(common_weights_iter_dims),
        /* bias_desc          */ generic_md(common_bias_dims),
        /* dst_layer_desc     */ formatted_md(dst_layer_dims,
                                                memory::format::tnc),
        /* dst_iter_desc      */ generic_md(dst_iter_dims)
    );
	
	// Describe primitive
    auto prim_desc
        = mkldnn::rnn_forward::primitive_desc(layer_desc, cpu_engine);
	
	 // Weights and biases, layer memory
    // Same layout should work across the layer, reordering
    // user memory to the RNN-friendly shapes.
	//weights_layer_memory
	auto common_weights_layer_memory = user_common_weights_layer_memory;
    primitive common_weights_layer_reorder;
    auto reorder_common_weights_layer = false;
    if (memory::primitive_desc(
            prim_desc.weights_layer_primitive_desc())
        != memory::primitive_desc(
            common_weights_layer_memory.get_primitive_desc())
    ) {
        common_weights_layer_memory
            = mkldnn::memory(prim_desc.weights_layer_primitive_desc());
        common_weights_layer_reorder
            = reorder(user_common_weights_layer_memory,
                        common_weights_layer_memory);
        reorder_common_weights_layer = true;
    }
	
	// weights_iter_memory
	auto common_weights_iter_memory = user_common_weights_iter_memory;
    primitive common_weights_iter_reorder;
    auto reorder_common_weights_iter = false;
    if (memory::primitive_desc(
            prim_desc.weights_iter_primitive_desc())
        != memory::primitive_desc(
            common_weights_iter_memory.get_primitive_desc())
    ) {
        common_weights_iter_memory
            = mkldnn::memory(prim_desc.weights_iter_primitive_desc());
        common_weights_iter_reorder
            = reorder(user_common_weights_iter_memory,
                        common_weights_iter_memory);
        reorder_common_weights_iter = true;
    }
	
	//bias_memory
    auto common_bias_memory = user_common_bias_memory;
    primitive common_bias_reorder;
    auto reorder_common_bias = false;
    if (memory::primitive_desc(
            prim_desc.bias_primitive_desc())
        != memory::primitive_desc(
            common_bias_memory.get_primitive_desc())
    ) {
        common_bias_memory
            = mkldnn::memory(prim_desc.bias_primitive_desc());
        common_bias_reorder
            = reorder(user_common_bias_memory,
                        common_bias_memory);
        reorder_common_bias = true;
    }

	auto dst_layer_memory = user_dst_layer_memory;
    primitive dst_layer_reorder;
    auto reorder_dst_layer = false;
    if (memory::primitive_desc(
            prim_desc.dst_layer_primitive_desc())
        != memory::primitive_desc(
            dst_layer_memory.get_primitive_desc())
    ) {
        dst_layer_memory
            = mkldnn::memory(prim_desc.dst_layer_primitive_desc());
        dst_layer_reorder
            = reorder(user_dst_layer_memory,
                       dst_layer_memory);
        reorder_dst_layer = true;
    }
	
	
	auto src_iter_memory
        = mkldnn::memory(prim_desc.src_iter_primitive_desc());	
	auto dst_iter_memory
        = mkldnn::memory(prim_desc.dst_iter_primitive_desc());
	 // We also create workspace memory based on the information from
    // the workspace_primitive_desc(). This is needed for internal
    // communication between forward and backward primitives during
    // training.
    // Inference mode doesn't need it, so initialize with null_memory_
	
    auto create_ws = [=](mkldnn::rnn_forward::primitive_desc &pd) {
        auto workspace_memory = null_memory_;
        if (is_training)
        {
            workspace_memory = mkldnn::memory(pd.workspace_primitive_desc());
        }
        return workspace_memory;
    };
	
	auto workspace_memory = create_ws(prim_desc);
		
	// Construct the RNN primitive objects
    rnn_forward rnn_layer = rnn_forward(
        /* aprimitive_desc */ prim_desc,
        /* src_layer       */ net_src_memory,
        /* src_iter        */ src_iter_memory,
        /* weights_layer   */ common_weights_layer_memory,
        /* weights_iter    */ common_weights_iter_memory,
        /* bias            */ common_bias_memory,
        /* dst_layer       */ dst_layer_memory,
        /* dst_iter        */ dst_iter_memory,
        /* workspace       */ workspace_memory
    );
	
	if (reorder_common_weights_layer)
        fwd_net.push_back(common_weights_layer_reorder);
	if (reorder_common_weights_iter)
        fwd_net.push_back(common_weights_iter_reorder);
    if (reorder_common_bias)
        fwd_net.push_back(common_bias_reorder);
    if (reorder_dst_layer)
        fwd_net.push_back(dst_layer_reorder);

    fwd_net.push_back(rnn_layer);
   
	
	/*----------------------------------------------------------------------*/
	/*----------------- Backward Stream -------------------------------------*/
	

	// We create the memory descriptors used by the user for backward 
	
	
	// User-provided memory for backward by data output
    std::vector<float> net_diff_src(
            tz_volume(net_src_dims),
            1.0f);
    auto diff_src_layer_memory
        = mkldnn::memory({ formatted_md(net_src_dims, memory::format::tnc),
                           cpu_engine }, net_diff_src.data());
						  

    std::vector<float> net_diff_dst(
        tz_volume(dst_layer_dims),
        1.0f);
    auto diff_dst_layer_memory
        = mkldnn::memory({ formatted_md(dst_layer_dims, memory::format::tnc),
                           cpu_engine }, net_diff_dst.data());
	
	// User-provided memory for backpropagation by weights
    std::vector<float> user_common_diff_weights_layer(
            tz_volume(common_weights_layer_dims),
            1.0f);
    auto user_common_diff_weights_layer_memory
        = mkldnn::memory({ formatted_md(common_weights_layer_dims,
                           memory::format::ldigo), cpu_engine },
                         user_common_diff_weights_layer.data());

    std::vector<float> user_common_diff_bias(
            tz_volume(common_bias_dims),
            1.0f);
    auto user_common_diff_bias_memory
        = mkldnn::memory({ formatted_md(common_bias_dims,
                           memory::format::ldgo), cpu_engine },
                         user_common_diff_bias.data());
	 
	 // Backward leftmost primitive descriptor
    rnn_backward::desc bwd_desc(
        /* aprop_kind              */ prop_kind::backward,
        /* cell                    */ uni_cell,
        /* direction               */ rnn_direction::unidirectional_left2right,
        /* src_layer_desc          */ formatted_md(net_src_dims, memory::format::tnc),
        /* src_iter_desc           */ generic_md(src_iter_dims),
        /* weights_layer_desc      */ generic_md(common_weights_layer_dims),
        /* weights_iter_desc       */ generic_md(common_weights_iter_dims),
        /* bias_desc               */ generic_md(common_bias_dims),
        /* dst_layer_desc          */ formatted_md(dst_layer_dims,
                                                    memory::format::tnc),
        /* dst_iter_desc           */ generic_md(dst_iter_dims),
        /* diff_src_layer_desc     */ generic_md(net_src_dims),
        /* diff_src_iter_desc      */ generic_md(src_iter_dims),
        /* diff_weights_layer_desc */ generic_md(common_weights_layer_dims),
        /* diff_weights_iter_desc  */ generic_md(common_weights_iter_dims),
        /* diff_bias_desc          */ generic_md(common_bias_dims),
        /* diff_dst_layer_desc     */ generic_md(dst_layer_dims),
        /* diff_dst_iter_desc      */ generic_md(dst_iter_dims)
    );
	auto bwd_prim_desc
        = mkldnn::rnn_backward::primitive_desc(
            bwd_desc, cpu_engine, prim_desc);
			
	auto src_layer_bwd_memory = net_src_memory;
	
	auto common_weights_layer_bwd_memory = common_weights_layer_memory;
    primitive common_weights_layer_bwd_reorder;
    auto reorder_common_weights_layer_bwd = false;
    if (memory::primitive_desc(
            bwd_prim_desc.weights_layer_primitive_desc())
        != memory::primitive_desc(
            prim_desc.weights_layer_primitive_desc())
    ) {
        common_weights_layer_bwd_memory
            = memory(bwd_prim_desc.weights_layer_primitive_desc());
        common_weights_layer_bwd_reorder
            = reorder(common_weights_layer_memory,
                        common_weights_layer_bwd_memory);
        reorder_common_weights_layer_bwd = true;
    }

    auto common_weights_iter_bwd_memory = common_weights_iter_memory;
    primitive common_weights_iter_bwd_reorder;
    auto reorder_common_weights_iter_bwd = false;
    if (memory::primitive_desc(
            bwd_prim_desc.weights_iter_primitive_desc())
        != memory::primitive_desc(
            prim_desc.weights_iter_primitive_desc())
    ) {
        common_weights_iter_bwd_memory
            = memory(bwd_prim_desc.weights_iter_primitive_desc());
        common_weights_iter_bwd_reorder
            = reorder(common_weights_iter_memory,
                        common_weights_iter_bwd_memory);
        reorder_common_weights_iter_bwd = true;
    }

    auto common_bias_bwd_memory = common_bias_memory;
    primitive common_bias_bwd_reorder;
    auto reorder_common_bias_bwd = false;
    if (memory::primitive_desc(
            bwd_prim_desc.bias_primitive_desc())
        != memory::primitive_desc(
            prim_desc.bias_primitive_desc())
    ) {
		cout<<"they are different common_bias_bwd_memory"<<endl;
        common_bias_bwd_memory
            = mkldnn::memory(bwd_prim_desc.bias_primitive_desc());
        common_bias_bwd_reorder
            = reorder(common_bias_memory,
                        common_bias_bwd_memory);
        reorder_common_bias_bwd = true;
    }

    // diff_weights and biases
    auto common_diff_weights_layer_memory
        = user_common_diff_weights_layer_memory;
    primitive common_diff_weights_layer_reorder;
    auto reorder_common_diff_weights_layer = false;
    if (memory::primitive_desc(
            bwd_prim_desc.diff_weights_layer_primitive_desc())
        != memory::primitive_desc(
            common_diff_weights_layer_memory.get_primitive_desc())
    ) {
        common_diff_weights_layer_memory
            = mkldnn::memory(
                bwd_prim_desc.diff_weights_layer_primitive_desc());
        common_diff_weights_layer_reorder
            = reorder(user_common_diff_weights_layer_memory,
                        common_diff_weights_layer_memory);
        reorder_common_diff_weights_layer = true;
    }

    auto common_diff_bias_memory = user_common_diff_bias_memory;
    primitive common_diff_bias_reorder;
    auto reorder_common_diff_bias = false;
    if (memory::primitive_desc(
            bwd_prim_desc.diff_bias_primitive_desc())
        != memory::primitive_desc(
            common_diff_bias_memory.get_primitive_desc())
    ) {
        common_diff_bias_memory
            = mkldnn::memory(bwd_prim_desc.diff_bias_primitive_desc());
        common_diff_bias_reorder
            = reorder(user_common_diff_bias_memory,
                        common_diff_bias_memory);
        reorder_common_diff_bias = true;
    }

    // dst_layer memory for backward pass
    auto dst_layer_bwd_memory = dst_layer_memory;
    primitive dst_layer_bwd_reorder;
    auto reorder_dst_layer_bwd = false;
    if (memory::primitive_desc(
            bwd_prim_desc.dst_layer_primitive_desc())
        != memory::primitive_desc(
            prim_desc.dst_layer_primitive_desc())
    ) {
        dst_layer_bwd_memory
            = mkldnn::memory(bwd_prim_desc.dst_layer_primitive_desc());
        dst_layer_bwd_reorder
            = reorder(dst_layer_memory,
                        dst_layer_bwd_memory);
        reorder_dst_layer_bwd = true;
    }
	
	auto dst_iter_bwd_memory = dst_iter_memory;
    primitive dst_iter_bwd_reorder;
    auto reorder_dst_iter_bwd = false;
    if (memory::primitive_desc(
            bwd_prim_desc.dst_iter_primitive_desc())
        != memory::primitive_desc(
            dst_iter_bwd_memory.get_primitive_desc())
    ) {
        dst_iter_bwd_memory
            = mkldnn::memory(bwd_prim_desc.dst_iter_primitive_desc());
        dst_iter_bwd_reorder
            = reorder(dst_iter_memory,
                        dst_iter_bwd_memory);
        reorder_dst_iter_bwd = true;
    }

    // Similar to forward, the backward primitives are connected
    // via "iter" parameters.
    auto common_diff_weights_iter_memory
        = mkldnn::memory(
            bwd_prim_desc.diff_weights_iter_primitive_desc());
    auto diff_dst_iter_memory
        = mkldnn::memory(bwd_prim_desc.diff_dst_iter_primitive_desc());
	auto diff_src_iter_memory
        = mkldnn::memory(bwd_prim_desc.diff_src_iter_primitive_desc());	
		
	// Construct the RNN primitive objects for backward
    rnn_backward layer_bwd = rnn_backward(
        /* aprimitive_desc    */ bwd_prim_desc,
        /* src_layer          */ src_layer_bwd_memory,
        /* src_iter           */ src_iter_memory,
        /* weights_layer      */ common_weights_layer_bwd_memory,
        /* weights_iter       */ common_weights_iter_bwd_memory,
        /* bias               */ common_bias_bwd_memory,
        /* dst_layer          */ dst_layer_bwd_memory,
        /* dst_iter           */ dst_iter_bwd_memory,
        /* diff_src_layer     */ diff_src_layer_memory,
        /* diff_src_iter      */ diff_src_iter_memory,
        /* diff_weights_layer */ common_diff_weights_layer_memory,
        /* diff_weights_iter  */ common_diff_weights_iter_memory,
        /* diff_bias          */ common_diff_bias_memory,
        /* diff_dst_layer     */ diff_dst_layer_memory,
        /* diff_dst_iter      */ diff_dst_iter_memory,
        /* workspace          */ workspace_memory
    );

	if (reorder_common_weights_layer_bwd)
        bwd_net.push_back(common_weights_layer_bwd_reorder);
    if (reorder_common_weights_iter_bwd)
        bwd_net.push_back(common_weights_iter_bwd_reorder);
    if (reorder_common_bias_bwd)
        bwd_net.push_back(common_bias_bwd_reorder);
    if (reorder_common_diff_weights_layer)
        bwd_net.push_back(common_diff_weights_layer_reorder);
    if (reorder_common_diff_bias)
        bwd_net.push_back(common_diff_bias_reorder);
	if (reorder_dst_iter_bwd)
        bwd_net.push_back(dst_iter_bwd_reorder);
    if (reorder_dst_layer_bwd)
        bwd_net.push_back(dst_layer_bwd_reorder);

    bwd_net.push_back(layer_bwd);
		stream(stream::kind::eager).submit(fwd_net).wait();
			stream(stream::kind::eager).submit(bwd_net).wait();
/*
	struct timeval start, end;
	for (int i =0; i < warmups + iters; i++) {
		if (i == warmups) {
			gettimeofday(&start, NULL);
			}
	    // Submit forward for execution
		stream(stream::kind::eager).submit(fwd_net).wait();
		
		if (is_training) 
			// Submit backward for execution
			stream(stream::kind::eager).submit(bwd_net).wait();
		
  }
  
  gettimeofday(&end, NULL);
  double t = (end.tv_sec - start.tv_sec) + (end.tv_usec -start.tv_usec) / 1000000.0;
  double avg_t = t / iters;
  std::cout << "\ttime " << avg_t << " s" << "\tSPS " << batch/ avg_t << std::endl;
*/
}


int main(int argc, char **argv) {
  if (argc != args_count) {
    std::ostringstream oss;
    oss << "Expected argument num " << args_count << ", got " << argc << std::endl;
    oss << "Usage: " << argv[0] << " [N] [T] [I] [H]";
      throw std::runtime_error(oss.str());
  }

  int index = 1;
  int N = std::atoi(argv[index++]);
  int T = std::atoi(argv[index++]);
  int I = std::atoi(argv[index++]);
  int H = std::atoi(argv[index++]);
  std::cout << "testing LSTM: N: " << N << " T: " << T << " I: "<< I << " H: " << H << std::endl;;

  try {
    run(T, N, I, H);
    run(T, N, I, H);
    run(T, N, I, H);
    run(T, N, I, H);
    run(T, N, I, H);
    run(T, N, I, H);
  } catch (error &e) {
    std::cerr << "status: " << e.status << std::endl;
    std::cerr << "message: " << e.message << std::endl;
    return 1;
  }
    return 0;
}	
