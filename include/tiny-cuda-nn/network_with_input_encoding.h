/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   network_with_input_encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A model that includes its encoding
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

TCNN_NAMESPACE_BEGIN

template <typename T>
class NetworkWithInputEncoding : public Network<float, T> {
public:
	NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims, const json& network) : m_encoding{encoding} {
		uint32_t alignment = network.contains("otype") && (equals_case_insensitive(network["otype"], "FullyFusedMLP") || equals_case_insensitive(network["otype"], "MegakernelMLP")) ? 16u : 8u;
		encoding->set_alignment(alignment);

		// Assume that row-major/SoA operations will be faster, so use it if supported.
		if (encoding->supports_output_layout(RM)) {
			encoding->set_output_layout(RM);
		}

		json local_network_config = network;
		local_network_config["n_input_dims"] = m_encoding->num_encoded_dims();
		local_network_config["n_output_dims"] = n_output_dims;
		m_network.reset(create_network<T>(local_network_config));
	}

	NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_output_dims, const json& encoding, const json& network)
	: NetworkWithInputEncoding{std::shared_ptr<Encoding<T>>{create_encoding<T>(n_dims_to_encode, encoding)}, n_output_dims, network} { }

	virtual ~NetworkWithInputEncoding() { }

	void inference(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<float>& output) override {
		GPUMatrixDynamic<T> network_input = {m_encoding->num_encoded_dims(), input.n(), stream, m_encoding->output_layout()};
		m_encoding->encode(stream, input.n(), {input.data(), input.m()}, {network_input.data(), network_input.m()}, nullptr, true);
		m_network->inference(stream, network_input, output);
	}

	void inference_mixed_precision(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices = true) override {
		GPUMatrixDynamic<T> network_input = {m_encoding->num_encoded_dims(), input.n(), stream, m_encoding->output_layout()};
		m_encoding->encode(stream, input.n(), {input.data(), input.m()}, {network_input.data(), network_input.m()}, nullptr, use_inference_matrices);
		m_network->inference_mixed_precision(stream, network_input, output, use_inference_matrices);
	}

	uint32_t num_encoded_dims() const {
		return m_encoding->num_encoded_dims();
	}

	void forward(cudaStream_t stream, const GPUMatrixDynamic<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();

		m_forward.network_input = GPUMatrixDynamic<T>{m_encoding->num_encoded_dims(), input.n(), stream, m_encoding->output_layout()};
		if (prepare_input_gradients) {
			m_forward.encoding_forward_gradient = GPUMatrix<float>{m_encoding->num_forward_gradient_dims(), input.n(), stream};
		}

		m_encoding->encode(
			stream,
			input.n(),
			{input.data(), input.m()},
			{m_forward.network_input.data(), m_forward.network_input.m()},
			prepare_input_gradients ? m_forward.encoding_forward_gradient.data() : nullptr,
			use_inference_matrices
		);
		m_network->forward(stream, m_forward.network_input, output, use_inference_matrices, prepare_input_gradients);
	}

	void forward_clear() override {
		m_forward.clear();
	}

	void backward(
		cudaStream_t stream,
		const GPUMatrixDynamic<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrixDynamic<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override {
		GPUMatrixDynamic<T> dL_dnetwork_input;
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = {m_encoding->num_encoded_dims(), input.n(), stream, m_encoding->output_layout()};
		}

		m_network->backward(stream, m_forward.network_input, output, dL_doutput, dL_dnetwork_input.data() ? &dL_dnetwork_input : nullptr, use_inference_matrices, compute_param_gradients);
		if (dL_dnetwork_input.data()) {
			m_encoding->backward(
				stream,
				input.n(),
				{dL_dnetwork_input.data(), dL_dnetwork_input.m()},
				dL_dinput ? m_forward.encoding_forward_gradient.data() : nullptr,
				dL_dinput ? PitchedPtr<float>{dL_dinput->data(), dL_dinput->m()} : PitchedPtr<float>{},
				{input.data(), input.m()}
			);
		}

		forward_clear();
	}

	void set_params(T* params, T* inference_params, T* backward_params, T* gradients) override {
		size_t offset = 0;
		m_network->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_network->n_params();

		m_encoding->set_params(
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset
		);
		offset += m_encoding->n_params();
	}

	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_network->n_params();

		m_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_encoding->n_params();
	}

	size_t n_params() const override {
		return m_encoding->n_params() + m_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return m_network->padded_output_width();
	}

	uint32_t output_width() const override {
		return m_network->output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return m_network->layer_sizes();
	}

	uint32_t width(uint32_t layer) const override {
		return layer == 0 ? m_encoding->num_encoded_dims() : m_network->width(layer - 1);
	}

	uint32_t num_forward_activations() const override {
		return m_network->num_forward_activations() + 1;
	}

	std::pair<const T*, MatrixLayout> forward_activations(uint32_t layer) const override {
		if (!m_forward.network_input.data()) {
			throw std::runtime_error{"Must call forward() before accessing activations."};
		}
		return layer == 0 ? std::make_pair<const T*, MatrixLayout>(m_forward.network_input.data(), m_encoding->output_layout()) : m_network->forward_activations(layer - 1);
	}

	uint32_t input_width() const {
		return m_encoding->num_dims_to_encode();
	}

	const std::shared_ptr<Encoding<T>>& encoding() const {
		return m_encoding.get();
	}

public:
	std::unique_ptr<Network<T>> m_network;
	std::shared_ptr<Encoding<T>> m_encoding;

	// Storage of forward pass data
	struct {
		GPUMatrixDynamic<T> network_input;
		GPUMatrix<float> encoding_forward_gradient;

		void clear() {
			network_input = GPUMatrixDynamic<T>{};
			encoding_forward_gradient = GPUMatrix<float>{};
		}
	} m_forward;
};

TCNN_NAMESPACE_END
