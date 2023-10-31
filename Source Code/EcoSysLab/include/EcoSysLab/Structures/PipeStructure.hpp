#pragma once

#include "ecosyslab_export.h"

namespace EcoSysLab
{
	typedef int PipeHandle;
	typedef int PipeNodeHandle;

	struct PipeNodeInfo
	{
		glm::vec3 m_globalPosition = glm::vec3(0.0f);
		glm::quat m_globalRotation = glm::vec3(0.0f);

		glm::vec2 m_localPosition = glm::vec2(0.0f);

		float m_thickness = 0.0f;
		
	};

	struct PipeInfo
	{
		glm::vec4 m_color = glm::vec4(1.0f);
	};

	template<typename PipeNodeData>
	class PipeNode
	{
		template<typename PGD, typename PD, typename PND>
		friend class PipeGroup;

		bool m_endNode = true;
		bool m_recycled = false;
		PipeNodeHandle m_prevHandle = -1;
		PipeNodeHandle m_handle = -1;
		PipeNodeHandle m_nextHandle = -1;
		
		PipeHandle m_pipeHandle = -1;

		int m_index = -1;
	public:
		PipeNodeData m_data;
		PipeNodeInfo m_info;

		/**
		 * Whether this node is the end node.
		 * @return True if this is end node, false else wise.
		 */
		[[nodiscard]] bool IsEndPipeNode() const;

		/**
		 * Whether this node is recycled (removed).
		 * @return True if this node is recycled (removed), false else wise.
		 */
		[[nodiscard]] bool IsRecycled() const;

		/**
		 * Get the handle of self.
		 * @return PipeNodeHandle of current node.
		 */
		[[nodiscard]] PipeNodeHandle GetHandle() const;

		/**
		 * Get the handle of belonged pipe.
		 * @return PipeHandle of current node.
		 */
		[[nodiscard]] PipeHandle GetPipeHandle() const;
		/**
		 * Get the handle of prev node.
		 * @return PipeNodeHandle of current node.
		 */
		[[nodiscard]] PipeNodeHandle GetPrevHandle() const;

		/**
		 * Get the handle of prev node.
		 * @return PipeNodeHandle of current node.
		 */
		[[nodiscard]] PipeNodeHandle GetNextHandle() const;

		[[nodiscard]] int GetIndex() const;

		explicit PipeNode(PipeHandle pipeHandle, PipeNodeHandle handle, PipeNodeHandle prevHandle);
	};

	template<typename PipeData>
	class Pipe
	{
		template<typename PGD, typename PD, typename PND>
		friend class PipeGroup;

		bool m_recycled = false;
		PipeHandle m_handle = -1;

		std::vector<PipeNodeHandle> m_nodeHandles;

	public:
		PipeData m_data;
		PipeInfo m_info;

		/**
		 * Whether this node is recycled (removed).
		 * @return True if this node is recycled (removed), false else wise.
		 */
		[[nodiscard]] bool IsRecycled() const;

		/**
		 * Get the handle of self.
		 * @return PipeNodeHandle of current node.
		 */
		[[nodiscard]] PipeHandle GetHandle() const;

		/**
		 * Access the nodes that belongs to this flow.
		 * @return The list of handles.
		 */
		[[nodiscard]] const std::vector<PipeNodeHandle>& PeekPipeNodeHandles() const;

		explicit Pipe(PipeHandle handle);
	};

	template<typename PipeGroupData, typename PipeData, typename PipeNodeData>
	class PipeGroup {

		std::vector<Pipe<PipeData>> m_pipes;
		std::vector<PipeNode<PipeNodeData>> m_pipeNodes;

		std::queue<PipeHandle> m_pipePool;
		std::queue<PipeNodeHandle> m_pipeNodePool;

		int m_version = -1;

		[[nodiscard]] PipeNodeHandle AllocatePipeNode(PipeHandle pipeHandle, PipeNodeHandle prevHandle, int index);
	public:
		PipeGroupData m_data;

		[[nodiscard]] PipeHandle AllocatePipe();

		/**
		 * Extend pipe during growth process. The flow structure will also be updated.
		 * @param targetHandle The handle of the node to branch/prolong
		 * @return The handle of new node.
		 */
		[[nodiscard]] PipeNodeHandle Extend(PipeHandle targetHandle);

		/**
		 * Insert pipe node during growth process. The flow structure will also be updated.
		 * @param targetHandle The handle of the pipe to be inserted.
		 * @param targetNodeHandle The handle of the pipe node to be inserted. If there's no subsequent node this will be a simple extend.
		 * @return The handle of new node.
		 */
		[[nodiscard]] PipeNodeHandle Insert(PipeHandle targetHandle, PipeNodeHandle targetNodeHandle);

		/**
		 * Recycle (Remove) a node, the descendents of this node will also be recycled. The relevant flow will also be removed/restructured.
		 * @param handle The handle of the node to be removed. Must be valid (non-zero and the node should not be recycled prior to this operation).
		 */
		void RecyclePipeNode(PipeNodeHandle handle);

		/**
		 * Recycle (Remove) a pipe. The relevant node will also be removed/restructured.
		 * @param handle The handle of the pipe to be removed. Must be valid (non-zero and the flow should not be recycled prior to this operation).
		 */
		void RecyclePipe(PipeHandle handle);


		[[nodiscard]] const std::vector<Pipe<PipeData>>& PeekPipes() const;

		[[nodiscard]] const std::vector<PipeNode<PipeNodeData>>& PeekPipeNodes() const;

		[[nodiscard]] std::vector<Pipe<PipeData>>& RefPipes();

		[[nodiscard]] std::vector<PipeNode<PipeNodeData>>& RefPipeNodes();

		[[nodiscard]] Pipe<PipeData>& RefPipe(PipeHandle handle);

		[[nodiscard]] PipeNode<PipeNodeData>& RefPipeNode(PipeNodeHandle handle);

		[[nodiscard]] const Pipe<PipeData>& PeekPipe(PipeHandle handle) const;

		[[nodiscard]] const PipeNode<PipeNodeData>& PeekPipeNode(PipeNodeHandle handle) const;

		/**
		 * Get the structural version of the tree. The version will change when the tree structure changes.
		 * @return The version
		 */
		[[nodiscard]] int GetVersion() const;
	};

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	PipeNodeHandle PipeGroup<PipeGroupData, PipeData, PipeNodeData>::AllocatePipeNode(PipeHandle pipeHandle, PipeNodeHandle prevHandle, const int index)
	{
		PipeNodeHandle newNodeHandle;
		if (m_pipeNodePool.empty()) {
			auto newNode = m_pipeNodes.emplace_back(pipeHandle, m_pipeNodes.size(), prevHandle);
			newNodeHandle = newNode.m_handle;
		}
		else {
			newNodeHandle = m_pipeNodePool.front();
			m_pipeNodePool.pop();
		}
		auto& node = m_pipeNodes[newNodeHandle];
		if (prevHandle != -1) {
			m_pipeNodes[prevHandle].m_nextHandle = newNodeHandle;
			m_pipeNodes[prevHandle].m_endNode = false;
			node.m_prevHandle = prevHandle;
		}
		node.m_pipeHandle = pipeHandle;
		node.m_index = index;
		node.m_recycled = false;
		return newNodeHandle;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	PipeHandle PipeGroup<PipeGroupData, PipeData, PipeNodeData>::AllocatePipe()
	{
		if (m_pipePool.empty()) {
			auto newPipe = m_pipes.emplace_back(m_pipes.size());
			m_version++;
			return newPipe.m_handle;
		}
		auto handle = m_pipePool.front();
		m_pipePool.pop();
		auto& pipe = m_pipes[handle];
		pipe.m_recycled = false;
		m_version++;
		return handle;
	}
	

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	PipeNodeHandle PipeGroup<PipeGroupData, PipeData, PipeNodeData>::Extend(PipeHandle targetHandle)
	{
		auto& pipe = m_pipes[targetHandle];
		assert(!pipe.m_recycled);
		auto prevHandle = -1;
		if (!pipe.m_nodeHandles.empty()) prevHandle = pipe.m_nodeHandles.back();
		const auto newNodeHandle = AllocatePipeNode(targetHandle, prevHandle, pipe.m_nodeHandles.size());
		pipe.m_nodeHandles.emplace_back(newNodeHandle);
		auto& node = m_pipeNodes[newNodeHandle];
		node.m_endNode = true;
		m_version++;
		return newNodeHandle;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	PipeNodeHandle PipeGroup<PipeGroupData, PipeData, PipeNodeData>::Insert(PipeHandle targetHandle, PipeNodeHandle targetNodeHandle)
	{
		auto& pipe = m_pipes[targetHandle];
		assert(!pipe.m_recycled);
		auto& prevNode = m_pipeNodes[targetNodeHandle];
		const auto prevNodeIndex = prevNode.m_index;
		const auto nextNodeHandle = pipe.m_nodeHandles[prevNodeIndex + 1];
		if (pipe.m_nodeHandles.size() - 1 == prevNodeIndex) return Extend(targetHandle);
		const auto newNodeHandle = AllocatePipeNode(targetHandle, targetNodeHandle, prevNodeIndex + 1);
		auto& newNode = m_pipeNodes[newNodeHandle];
		newNode.m_endNode = false;
		newNode.m_nextHandle = nextNodeHandle;
		auto& nextNode = m_pipeNodes[nextNodeHandle];
		nextNode.m_prevHandle = newNodeHandle;
		pipe.m_nodeHandles.insert(pipe.m_nodeHandles.begin() + prevNodeIndex + 1, newNodeHandle);
		for(int i = prevNodeIndex + 2; i < pipe.m_nodeHandles.size(); ++i)
		{
			m_pipeNodes[pipe.m_nodeHandles[i]].m_index = i;
		}
		m_version++;
		return newNodeHandle;
	}


	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	void PipeGroup<PipeGroupData, PipeData, PipeNodeData>::RecyclePipeNode(PipeNodeHandle handle)
	{
		//Recycle subsequent nodes from pipe.
		auto& node = m_pipeNodes[handle];
		assert(!node.m_recycled);
		if (node.m_nextHandle != -1)
		{
			RecyclePipeNode(node.m_nextHandle);
		}
		if (node.m_prevHandle != -1)
		{
			m_pipeNodes[node.m_prevHandle].m_nextHandle = -1;
			m_pipeNodes[node.m_prevHandle].m_endNode = true;
		}

		auto& pipe = m_pipes[node.m_pipeHandle];
		assert(pipe.m_nodeHandles.back() == handle);
		pipe.m_nodeHandles.pop_back();

		node.m_recycled = true;
		node.m_endNode = true;
		node.m_prevHandle = node.m_nextHandle = -1;
		node.m_data = {};
		node.m_info = {};

		node.m_index = -1;
		node.m_pipeHandle = -1;
		m_pipeNodePool.emplace(handle);
		m_version++;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	void PipeGroup<PipeGroupData, PipeData, PipeNodeData>::RecyclePipe(PipeHandle handle)
	{
		//Recycle all nodes;
		auto& pipe = m_pipes[handle];
		assert(!pipe.m_recycled);
		for (const auto& nodeHandle : pipe.m_nodeHandles)
		{
			auto& node = m_pipeNodes[nodeHandle];
			node.m_recycled = true;
			node.m_endNode = true;
			node.m_prevHandle = node.m_nextHandle = -1;
			node.m_data = {};
			node.m_info = {};

			node.m_index = -1;
			node.m_pipeHandle = -1;
			m_pipeNodePool.emplace(nodeHandle);
		}
		pipe.m_nodeHandles.clear();

		//Recycle pipe.
		pipe.m_recycled = true;
		pipe.m_data = {};
		pipe.m_info = {};
		m_pipePool.emplace(handle);
		m_version++;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	const std::vector<Pipe<PipeData>>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::PeekPipes() const
	{
		return m_pipes;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	const std::vector<PipeNode<PipeNodeData>>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::PeekPipeNodes() const
	{
		return m_pipeNodes;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	std::vector<Pipe<PipeData>>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::RefPipes()
	{
		return m_pipes;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	std::vector<PipeNode<PipeNodeData>>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::RefPipeNodes()
	{
		return m_pipeNodes;
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	Pipe<PipeData>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::RefPipe(PipeHandle handle)
	{
		return m_pipes[handle];
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	PipeNode<PipeNodeData>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::RefPipeNode(PipeNodeHandle handle)
	{
		return m_pipeNodes[handle];
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	const Pipe<PipeData>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::PeekPipe(PipeHandle handle) const
	{
		return m_pipes[handle];
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	const PipeNode<PipeNodeData>& PipeGroup<PipeGroupData, PipeData, PipeNodeData>::PeekPipeNode(
		PipeNodeHandle handle) const
	{
		return m_pipeNodes[handle];
	}

	template <typename PipeGroupData, typename PipeData, typename PipeNodeData>
	int PipeGroup<PipeGroupData, PipeData, PipeNodeData>::GetVersion() const
	{
		return m_version;
	}

	template <typename PipeNodeData>
	bool PipeNode<PipeNodeData>::IsEndPipeNode() const
	{
		return m_endNode;
	}

	template <typename PipeNodeData>
	bool PipeNode<PipeNodeData>::IsRecycled() const
	{
		return m_recycled;
	}

	template <typename PipeNodeData>
	PipeNodeHandle PipeNode<PipeNodeData>::GetHandle() const
	{
		return m_handle;
	}

	template <typename PipeNodeData>
	PipeHandle PipeNode<PipeNodeData>::GetPipeHandle() const
	{
		return m_pipeHandle;
	}

	template <typename PipeNodeData>
	PipeNodeHandle PipeNode<PipeNodeData>::GetPrevHandle() const
	{
		return m_prevHandle;
	}

	template <typename PipeNodeData>
	PipeNodeHandle PipeNode<PipeNodeData>::GetNextHandle() const
	{
		return m_nextHandle;
	}

	template <typename PipeNodeData>
	int PipeNode<PipeNodeData>::GetIndex() const
	{
		return m_index;
	}

	template <typename PipeNodeData>
	PipeNode<PipeNodeData>::PipeNode(const PipeHandle pipeHandle, const PipeNodeHandle handle, const PipeNodeHandle prevHandle)
	{
		m_pipeHandle = pipeHandle;
		m_handle = handle;
		m_prevHandle = prevHandle;
		m_nextHandle = -1;
		m_recycled = false;
		m_endNode = true;

		m_index = -1;
		m_data = {};
		m_info = {};
	}

	template <typename PipeData>
	bool Pipe<PipeData>::IsRecycled() const
	{
		return m_recycled;
	}

	template <typename PipeData>
	PipeHandle Pipe<PipeData>::GetHandle() const
	{
		return m_handle;
	}

	template <typename PipeData>
	const std::vector<PipeNodeHandle>& Pipe<PipeData>::PeekPipeNodeHandles() const
	{
		return m_nodeHandles;
	}

	template <typename PipeData>
	Pipe<PipeData>::Pipe(const PipeHandle handle)
	{
		m_handle = handle;
		m_recycled = false;

		m_nodeHandles.clear();

		m_data = {};
		m_info = {};
	}
}
