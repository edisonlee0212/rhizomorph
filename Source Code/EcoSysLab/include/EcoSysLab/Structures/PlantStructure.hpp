#pragma once

#include "ecosyslab_export.h"

namespace EcoSysLab {
    typedef int NodeHandle;
    typedef int FlowHandle;

#pragma region Structural Info
    struct NodeInfo {
        glm::vec3 m_globalPosition = glm::vec3(0.0f);
        glm::quat m_globalRotation = glm::vec3(0.0f);

        float m_length = 0.0f;
        float m_thickness = 0.1f;
        glm::quat m_localRotation = glm::vec3(0.0f);
        glm::vec3 m_localPosition = glm::vec3(0.0f);

        glm::quat m_regulatedGlobalRotation = glm::vec3(0.0f);
    };

    struct FlowInfo {
        glm::vec3 m_globalStartPosition = glm::vec3(0.0f);
        glm::quat m_globalStartRotation = glm::vec3(0.0f);
        float m_startThickness = 0.0f;

        glm::vec3 m_globalEndPosition = glm::vec3(0.0f);
        glm::quat m_globalEndRotation = glm::vec3(0.0f);
        float m_endThickness = 0.0f;
    };
#pragma endregion

    template<typename NodeData>
    class Node {
#pragma region Private

        template<typename FD>
        friend class Flow;

        template<typename SD, typename FD, typename ID>
        friend class Skeleton;

        bool m_flowStartNode = true;
        bool m_endNode = true;
        bool m_recycled = false;
        NodeHandle m_handle = -1;
        FlowHandle m_flowHandle = -1;
        NodeHandle m_parentHandle = -1;
        std::vector<NodeHandle> m_childHandles;
        bool m_apical = true;
#pragma endregion
    public:
        NodeData m_data;
        /**
         * The structural information of current node.
         */
        NodeInfo m_info;

        /**
         * Whether this node is the start node.
         * @return True if this is start node, false else wise.
         */
        [[nodiscard]] bool IsFlowStartNode() const;

        /**
         * Whether this node is the end node.
         * @return True if this is end node, false else wise.
         */
        [[nodiscard]] bool IsEndNode() const;

        /**
         * Whether this node is recycled (removed).
         * @return True if this node is recycled (removed), false else wise.
         */
        [[nodiscard]] bool IsRecycled() const;
        /**
         * Whether this node is apical.
         * @return True if this node is apical, false else wise.
         */
        [[nodiscard]] bool IsApical() const;
        /**
         * Get the handle of self.
         * @return NodeHandle of current node.
         */
        [[nodiscard]] NodeHandle GetHandle() const;

        /**
         * Get the handle of parent.
         * @return NodeHandle of parent node.
         */
        [[nodiscard]] NodeHandle GetParentHandle() const;

        /**
         * Get the handle to belonged flow.
         * @return FlowHandle of belonged flow.
         */
        [[nodiscard]] FlowHandle GetFlowHandle() const;

        /**
         * Access the children by their handles.
         * @return The list of handles.
         */
        [[nodiscard]] const std::vector<NodeHandle> &RefChildHandles() const;
        Node();
        Node(NodeHandle handle);
    };

    template<typename FlowData>
    class Flow {
#pragma region Private

        template<typename SD, typename FD, typename ID>
        friend class Skeleton;

        bool m_recycled = false;
        FlowHandle m_handle = -1;
        std::vector<NodeHandle> m_nodes;
        FlowHandle m_parentHandle = -1;
        std::vector<FlowHandle> m_childHandles;
        bool m_apical = false;
#pragma endregion
    public:
        FlowData m_data;
        FlowInfo m_info;

        /**
         * Whether this flow is recycled (removed).
         * @return True if this flow is recycled (removed), false else wise.
         */
        [[nodiscard]] bool IsRecycled() const;

        /**
         * Whether this flow is extended from an apical bud. The apical flow will have the same order as parent flow.
         * @return True if this flow is from apical bud.
         */
        [[nodiscard]] bool IsApical() const;

        /**
         * Get the handle of self.
         * @return FlowHandle of current flow.
         */
        [[nodiscard]] FlowHandle GetHandle() const;

        /**
         * Get the handle of parent.
         * @return FlowHandle of parent flow.
         */
        [[nodiscard]] FlowHandle GetParentHandle() const;

        /**
         * Access the children by their handles.
         * @return The list of handles.
         */
        [[nodiscard]] const std::vector<FlowHandle> &RefChildHandles() const;

        /**
         * Access the nodes that belongs to this flow.
         * @return The list of handles.
         */
        [[nodiscard]] const std::vector<NodeHandle> &RefNodeHandles() const;

        explicit Flow(FlowHandle handle);
    };

    template<typename SkeletonData, typename FlowData, typename NodeData>
    class Skeleton {
#pragma region Private
        std::vector<Flow<FlowData>> m_flows;
        std::vector<Node<NodeData>> m_nodes;
        std::queue<NodeHandle> m_nodePool;
        std::queue<FlowHandle> m_flowPool;

        int m_newVersion = 0;
        int m_version = -1;
        std::vector<NodeHandle> m_sortedNodeList;
        std::vector<FlowHandle> m_sortedFlowList;

        NodeHandle AllocateNode();

        void RecycleNodeSingle(NodeHandle handle, const std::function<void(NodeHandle)>& nodeHandler);

        void RecycleFlowSingle(FlowHandle handle, const std::function<void(FlowHandle)>& flowHandler);

        FlowHandle AllocateFlow();

        void SetParentFlow(FlowHandle targetHandle, FlowHandle parentHandle);

        void DetachChildFlow(FlowHandle targetHandle, FlowHandle childHandle);

        void SetParentNode(NodeHandle targetHandle, NodeHandle parentHandle);

        void DetachChildNode(NodeHandle targetHandle, NodeHandle childHandle);

#pragma endregion
    public:
        template<typename SrcSkeletonData, typename SrcFlowData, typename SrcNodeData>
        void Clone(const Skeleton<SrcSkeletonData, SrcFlowData, SrcNodeData>& srcSkeleton, const std::function<void(NodeHandle srcNodeHandle, NodeHandle dstNodeHandle)>& postProcess);


        SkeletonData m_data;

        /**
         * Recycle (Remove) a node, the descendents of this node will also be recycled. The relevant flow will also be removed/restructured.
         * @param handle The handle of the node to be removed. Must be valid (non-zero and the node should not be recycled prior to this operation).
         * @param flowHandler Function to be called right before a flow in recycled.
         * @param nodeHandler Function to be called right before a node in recycled.
         */
        void RecycleNode(NodeHandle handle,
            const std::function<void(FlowHandle)>& flowHandler,
            const std::function<void(NodeHandle)>& nodeHandler);

        /**
         * Recycle (Remove) a flow, the descendents of this flow will also be recycled. The relevant node will also be removed/restructured.
         * @param handle The handle of the flow to be removed. Must be valid (non-zero and the flow should not be recycled prior to this operation).
         * @param flowHandler Function to be called right before a flow in recycled.
         * @param nodeHandler Function to be called right before a node in recycled.
         */
        void RecycleFlow(FlowHandle handle, 
            const std::function<void(FlowHandle)>& flowHandler,
            const std::function<void(NodeHandle)>& nodeHandler);

        /**
         * Branch/prolong node during growth process. The flow structure will also be updated.
         * @param targetHandle The handle of the node to branch/prolong
         * @param branching True if branching, false if prolong. During branching, 2 new flows will be generated.
         * @param breakFlow Whether we need to create a new flow
         * @return The handle of new node.
         */
        [[nodiscard]] NodeHandle Extend(NodeHandle targetHandle, bool branching, bool breakFlow = false);

        /**
         * To retrieve a list of handles of all nodes contained within the tree.
         * @return The list of handles of nodes sorted from root to ends.
         */
        [[nodiscard]] const std::vector<NodeHandle> &RefSortedNodeList() const;

        /**
         * To retrieve a list of handles of all flows contained within the tree.
         * @return The list of handles of flows sorted from root to ends.
         */
        [[nodiscard]] const std::vector<FlowHandle> &RefSortedFlowList() const;

        [[nodiscard]] const std::vector<Flow<FlowData>> &RefRawFlows() const;

        [[nodiscard]] const std::vector<Node<NodeData>> &RefRawNodes() const;

        /**
         *  Force the structure to sort the node and flow list.
         *  \n!!You MUST call this after you prune the tree or altered the tree structure manually!!
         */
        void SortLists();

        Skeleton();

        /**
         * Get the structural version of the tree. The version will change when the tree structure changes.
         * @return The version
         */
        [[nodiscard]] int GetVersion() const;

        /**
         * Calculate the structural information of the flows.
         */
        void CalculateFlows();

        /* 
         * Calculate the global transforms for nodes based on relative information.
         */
        void CalculateTransforms();
        /**
         * Retrieve a modifiable reference to the node with the handle.
         * @param handle The handle to the target node.
         * @return The modifiable reference to the node.
         */
        Node<NodeData> &RefNode(NodeHandle handle);

        /**
         * Retrieve a modifiable reference to the flow with the handle.
         * @param handle The handle to the target flow.
         * @return The modifiable reference to the flow.
         */
        Flow<FlowData> &RefFlow(FlowHandle handle);

        /**
         * Retrieve a non-modifiable reference to the node with the handle.
         * @param handle The handle to the target node.
         * @return The non-modifiable reference to the node.
         */
        [[nodiscard]] const Node<NodeData> &PeekNode(NodeHandle handle) const;

        /**
         * Retrieve a non-modifiable reference to the flow with the handle.
         * @param handle The handle to the target flow.
         * @return The non-modifiable reference to the flow.
         */
        [[nodiscard]] const Flow<FlowData> &PeekFlow(FlowHandle handle) const;

        /**
         * The min value of the bounding box of current tree structure.
         */
        glm::vec3 m_min = glm::vec3(0.0f);

        /**
         * The max value of the bounding box of current tree structure.
         */
        glm::vec3 m_max = glm::vec3(0.0f);
    };

    struct BaseSkeletonData{};
    struct BaseFlowData {};
    struct BaseNodeData {};

    typedef Skeleton<BaseSkeletonData, BaseFlowData, BaseNodeData> BaseSkeleton;

#pragma region TreeSkeleton
#pragma region Helper

    template<typename SkeletonData, typename FlowData, typename NodeData>
    Flow<FlowData> &Skeleton<SkeletonData, FlowData, NodeData>::RefFlow(FlowHandle handle) {
        assert(handle >= 0 && handle < m_flows.size());
        return m_flows[handle];
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    const Flow<FlowData> &Skeleton<SkeletonData, FlowData, NodeData>::PeekFlow(FlowHandle handle) const {
        assert(handle >= 0 && handle < m_flows.size());
        return m_flows[handle];
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    Node<NodeData> &
    Skeleton<SkeletonData, FlowData, NodeData>::RefNode(NodeHandle handle) {
        assert(handle >= 0 && handle < m_nodes.size());
        return m_nodes[handle];
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    const Node<NodeData> &
    Skeleton<SkeletonData, FlowData, NodeData>::PeekNode(NodeHandle handle) const {
        assert(handle >= 0 && handle < m_nodes.size());
        return m_nodes[handle];
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::SortLists() {
        if (m_version == m_newVersion) return;
        if (m_nodes.empty()) return;
        m_version = m_newVersion;
        m_sortedFlowList.clear();
        std::queue<FlowHandle> flowWaitList;
        flowWaitList.push(0);
        while (!flowWaitList.empty()) {
            m_sortedFlowList.emplace_back(flowWaitList.front());
            flowWaitList.pop();
            for (const auto &i: m_flows[m_sortedFlowList.back()].m_childHandles) {
                flowWaitList.push(i);
            }

        }

        m_sortedNodeList.clear();
        std::queue<NodeHandle> nodeWaitList;
        nodeWaitList.push(0);
        while (!nodeWaitList.empty()) {
            m_sortedNodeList.emplace_back(nodeWaitList.front());
            nodeWaitList.pop();
            for (const auto &i: m_nodes[m_sortedNodeList.back()].m_childHandles) {
                nodeWaitList.push(i);
            }
        }
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    const std::vector<FlowHandle> &Skeleton<SkeletonData, FlowData, NodeData>::RefSortedFlowList() const {
        return m_sortedFlowList;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    const std::vector<NodeHandle> &
    Skeleton<SkeletonData, FlowData, NodeData>::RefSortedNodeList() const {
        return m_sortedNodeList;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    NodeHandle
    Skeleton<SkeletonData, FlowData, NodeData>::Extend(NodeHandle targetHandle, const bool branching, const bool breakFlow) {
        assert(targetHandle < m_nodes.size());
        auto &targetNode = m_nodes[targetHandle];
        assert(!targetNode.m_recycled);
        assert(targetNode.m_flowHandle < m_flows.size());
        auto &flow = m_flows[targetNode.m_flowHandle];
        assert(!flow.m_recycled);
        auto newNodeHandle = AllocateNode();
        SetParentNode(newNodeHandle, targetHandle);
        auto &originalNode = m_nodes[targetHandle];
        auto &newNode = m_nodes[newNodeHandle];
        originalNode.m_endNode = false;
        if (branching) {
            auto newFlowHandle = AllocateFlow();
            auto &newFlow = m_flows[newFlowHandle];

            newNode.m_flowHandle = newFlowHandle;
            newNode.m_apical = false;
            newFlow.m_nodes.emplace_back(newNodeHandle);
            newFlow.m_apical = false;
            newNode.m_flowStartNode = true;
            if (targetHandle != m_flows[originalNode.m_flowHandle].m_nodes.back()) {
                auto extendedFlowHandle = AllocateFlow();
                auto &extendedFlow = m_flows[extendedFlowHandle];
                extendedFlow.m_apical = true;
                //Find target node.
                auto &originalFlow = m_flows[originalNode.m_flowHandle];
                for (auto r = originalFlow.m_nodes.begin(); r != originalFlow.m_nodes.end(); ++r) {
                    if (*r == targetHandle) {
                        extendedFlow.m_nodes.insert(extendedFlow.m_nodes.end(), r + 1,
                                                    originalFlow.m_nodes.end());
                        originalFlow.m_nodes.erase(r + 1, originalFlow.m_nodes.end());
                        break;
                    }
                }
                for (const auto &extractedNodeHandle: extendedFlow.m_nodes) {
                    auto &extractedNode = m_nodes[extractedNodeHandle];
                    extractedNode.m_flowHandle = extendedFlowHandle;
                }
                extendedFlow.m_childHandles = originalFlow.m_childHandles;
                originalFlow.m_childHandles.clear();
                for (const auto &childFlowHandle: extendedFlow.m_childHandles) {
                    m_flows[childFlowHandle].m_parentHandle = extendedFlowHandle;
                }
                SetParentFlow(extendedFlowHandle, originalNode.m_flowHandle);
            }
            SetParentFlow(newFlowHandle, originalNode.m_flowHandle);
        }
    	else if(breakFlow)
        {
            auto newFlowHandle = AllocateFlow();
            auto& newFlow = m_flows[newFlowHandle];
            newNode.m_flowHandle = newFlowHandle;
            newNode.m_apical = true;
            newNode.m_flowStartNode = true;
            newFlow.m_nodes.emplace_back(newNodeHandle);
            newFlow.m_apical = true;
            SetParentFlow(newFlowHandle, originalNode.m_flowHandle);
        }
    	else {
            flow.m_nodes.emplace_back(newNodeHandle);
            newNode.m_flowHandle = originalNode.m_flowHandle;
            newNode.m_flowStartNode = false;
            newNode.m_apical = true;
        }
        m_newVersion++;
        return newNodeHandle;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::RecycleFlow(FlowHandle handle, 
        const std::function<void(FlowHandle)>& flowHandler,
        const std::function<void(NodeHandle)>& nodeHandler) {
        assert(handle != 0);
        assert(!m_flows[handle].m_recycled);
        auto &flow = m_flows[handle];
        //Remove children
        auto children = flow.m_childHandles;
        for (const auto &child: children) {
            if (m_flows[child].m_recycled) continue;
            RecycleFlow(child, flowHandler, nodeHandler);
        }
        //Detach from parent
        auto parentHandle = flow.m_parentHandle;
        if (parentHandle != -1) DetachChildFlow(parentHandle, handle);
        //Remove node
        if (!flow.m_nodes.empty()) {
            //Detach first node from parent.
            auto nodes = flow.m_nodes;
            for (const auto &i: nodes) {
                RecycleNodeSingle(i, nodeHandler);
            }
        }
        RecycleFlowSingle(handle, flowHandler);
        m_newVersion++;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::RecycleNode(NodeHandle handle,
        const std::function<void(FlowHandle)>& flowHandler,
        const std::function<void(NodeHandle)>& nodeHandler) {
        assert(handle != 0);
        assert(!m_nodes[handle].m_recycled);
        auto &node = m_nodes[handle];
        auto flowHandle = node.m_flowHandle;
        auto &flow = m_flows[flowHandle];
        if (handle == flow.m_nodes[0]) {
            auto parentFlowHandle = flow.m_parentHandle;
            
            RecycleFlow(node.m_flowHandle, flowHandler, nodeHandler);
            //Connect parent branch with the only apical child flow.
            auto &parentFlow = m_flows[parentFlowHandle];
            if (parentFlow.m_childHandles.size() == 1) {
                auto childHandle = parentFlow.m_childHandles[0];
                auto &childFlow = m_flows[childHandle];
                if (childFlow.m_apical) {
                    for (const auto &nodeHandle: childFlow.m_nodes) {
                        m_nodes[nodeHandle].m_flowHandle = parentFlowHandle;
                    }
                    for (const auto &grandChildFlowHandle: childFlow.m_childHandles) {
                        m_flows[grandChildFlowHandle].m_parentHandle = parentFlowHandle;
                    }
                    parentFlow.m_nodes.insert(parentFlow.m_nodes.end(), childFlow.m_nodes.begin(),
                                              childFlow.m_nodes.end());
                    parentFlow.m_childHandles.clear();
                    parentFlow.m_childHandles.insert(parentFlow.m_childHandles.end(),
                                                     childFlow.m_childHandles.begin(),
                                                     childFlow.m_childHandles.end());
                    RecycleFlowSingle(childHandle, flowHandler);
                }
            }
            return;
        }
        //Collect list of subsequent nodes
        std::vector<NodeHandle> subsequentNodes;
        while (flow.m_nodes.back() != handle) {
            subsequentNodes.emplace_back(flow.m_nodes.back());
            flow.m_nodes.pop_back();
        }
        subsequentNodes.emplace_back(flow.m_nodes.back());
        flow.m_nodes.pop_back();
        assert(!flow.m_nodes.empty());
        //Detach from parent
        if (node.m_parentHandle != -1) DetachChildNode(node.m_parentHandle, handle);
        //From end node remove one by one.
        NodeHandle prev = -1;
        for (const auto &i: subsequentNodes) {
            auto children = m_nodes[i].m_childHandles;
            for (const auto &childNodeHandle: children) {
                if (childNodeHandle == prev) continue;
                auto &child = m_nodes[childNodeHandle];
                assert(!child.m_recycled);
                auto childBranchHandle = child.m_flowHandle;
                if (childBranchHandle != flowHandle) {
                    RecycleFlow(childBranchHandle, flowHandler, nodeHandler);
                }
            }
            prev = i;
            RecycleNodeSingle(i, nodeHandler);

        }
        m_newVersion++;
    }

#pragma endregion
#pragma region Internal

    template<typename NodeData>
    Node<NodeData>::Node(const NodeHandle handle) {
        m_handle = handle;
        m_recycled = false;
        m_endNode = true;
        m_data = {};
        m_info = {};
    }

    template <typename NodeData>
    bool Node<NodeData>::IsFlowStartNode() const
    {
	    return m_flowStartNode;
    }

    template<typename NodeData>
    bool Node<NodeData>::IsEndNode() const {
        return m_endNode;
    }

    template<typename NodeData>
    bool Node<NodeData>::IsRecycled() const {
        return m_recycled;
    }

    template <typename NodeData>
    bool Node<NodeData>::IsApical() const
    {
        return m_apical;
    }

    template<typename NodeData>
    NodeHandle Node<NodeData>::GetHandle() const {
        return m_handle;
    }

    template<typename NodeData>
    NodeHandle Node<NodeData>::GetParentHandle() const {
        return m_parentHandle;
    }

    template<typename NodeData>
    FlowHandle Node<NodeData>::GetFlowHandle() const {
        return m_flowHandle;
    }

    template<typename NodeData>
    const std::vector<NodeHandle> &Node<NodeData>::RefChildHandles() const {
        return m_childHandles;
    }

    template <typename NodeData>
    Node<NodeData>::Node()
    {
        m_handle = -1;
        m_recycled = false;
        m_endNode = true;
        m_data = {};
        m_info = {};
    }

    template<typename FlowData>
    Flow<FlowData>::Flow(const FlowHandle handle) {
        m_handle = handle;
        m_recycled = false;
        m_data = {};
        m_info = {};
        m_apical = false;
    }

    template<typename FlowData>
    const std::vector<NodeHandle> &Flow<FlowData>::RefNodeHandles() const {
        return m_nodes;
    }

    template<typename FlowData>
    FlowHandle Flow<FlowData>::GetParentHandle() const {
        return m_parentHandle;
    }

    template<typename FlowData>
    const std::vector<FlowHandle> &Flow<FlowData>::RefChildHandles() const {
        return m_childHandles;
    }

    template<typename FlowData>
    bool Flow<FlowData>::IsRecycled() const {
        return m_recycled;
    }

    template<typename FlowData>
    FlowHandle Flow<FlowData>::GetHandle() const { return m_handle; }

    template<typename FlowData>
    bool Flow<FlowData>::IsApical() const {
        return m_apical;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    Skeleton<SkeletonData, FlowData, NodeData>::Skeleton() {
        AllocateFlow();
        AllocateNode();
        auto &rootBranch = m_flows[0];
        auto &rootNode = m_nodes[0];
        rootNode.m_flowHandle = 0;
        rootBranch.m_nodes.emplace_back(0);
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::DetachChildNode(NodeHandle targetHandle,
                                                                     NodeHandle childHandle) {
        assert(targetHandle >= 0 && childHandle >= 0 && targetHandle < m_nodes.size() &&
               childHandle < m_nodes.size());
        auto &targetNode = m_nodes[targetHandle];
        auto &childNode = m_nodes[childHandle];
        assert(!targetNode.m_recycled);
        assert(!childNode.m_recycled);
        auto &children = targetNode.m_childHandles;
        for (int i = 0; i < children.size(); i++) {
            if (children[i] == childHandle) {
                children[i] = children.back();
                children.pop_back();
                childNode.m_parentHandle = -1;
                if (children.empty()) targetNode.m_endNode = true;
                return;
            }
        }
    }

    template <typename SkeletonData, typename FlowData, typename NodeData>
    template <typename SrcSkeletonData, typename SrcFlowData, typename SrcNodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::Clone(
	    const Skeleton<SrcSkeletonData, SrcFlowData, SrcNodeData>& srcSkeleton, const std::function<void(NodeHandle srcNodeHandle, NodeHandle dstNodeHandle)>& postProcess)
    {
        
        m_data = {};
        m_nodes.clear();
        m_flows.clear();
        m_flowPool = {};
        m_nodePool = {};
        m_sortedNodeList.clear();
        m_sortedFlowList.clear();

        AllocateFlow();
        auto firstNodeHandle = AllocateNode();
        auto& rootBranch = m_flows[0];
        auto& rootNode = m_nodes[firstNodeHandle];
        rootNode.m_flowHandle = 0;
        rootBranch.m_nodes.emplace_back(0);

        const auto& sortedSrcNodeList = srcSkeleton.RefSortedNodeList();
        std::unordered_map<NodeHandle, NodeHandle> nodeHandleMap;
        nodeHandleMap[0] = 0;
        for (const auto& srcNodeHandle : sortedSrcNodeList)
        {
            if (srcNodeHandle == 0)
            {
                m_nodes[0].m_info = srcSkeleton.PeekNode(0).m_info;
                postProcess(0, 0);
            }
            else
            {
                const auto& srcNode = srcSkeleton.PeekNode(srcNodeHandle);
                auto dstNodeHandle = this->Extend(nodeHandleMap.at(srcNode.GetParentHandle()), !srcNode.IsApical(), false);
                nodeHandleMap[srcNodeHandle] = dstNodeHandle;
                m_nodes[dstNodeHandle].m_info = srcNode.m_info;
                postProcess(srcNodeHandle, dstNodeHandle);
            }
        }
        SortLists();
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::SetParentNode(NodeHandle targetHandle,
                                                                   NodeHandle parentHandle) {
        assert(targetHandle >= 0 && parentHandle >= 0 && targetHandle < m_nodes.size() &&
               parentHandle < m_nodes.size());
        auto &targetNode = m_nodes[targetHandle];
        auto &parentNode = m_nodes[parentHandle];
        assert(!targetNode.m_recycled);
        assert(!parentNode.m_recycled);
        targetNode.m_parentHandle = parentHandle;
        parentNode.m_childHandles.emplace_back(targetHandle);
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void
    Skeleton<SkeletonData, FlowData, NodeData>::DetachChildFlow(FlowHandle targetHandle,
                                                                FlowHandle childHandle) {
        assert(targetHandle >= 0 && childHandle >= 0 && targetHandle < m_flows.size() &&
               childHandle < m_flows.size());
        auto &targetBranch = m_flows[targetHandle];
        auto &childBranch = m_flows[childHandle];
        assert(!targetBranch.m_recycled);
        assert(!childBranch.m_recycled);

        if (!childBranch.m_nodes.empty()) {
            auto firstNodeHandle = childBranch.m_nodes[0];
            auto &firstNode = m_nodes[firstNodeHandle];
            if (firstNode.m_parentHandle != -1)
                DetachChildNode(firstNode.m_parentHandle, firstNodeHandle);
        }

        auto &children = targetBranch.m_childHandles;
        for (int i = 0; i < children.size(); i++) {
            if (children[i] == childHandle) {
                children[i] = children.back();
                children.pop_back();
                childBranch.m_parentHandle = -1;
                return;
            }
        }
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void
    Skeleton<SkeletonData, FlowData, NodeData>::SetParentFlow(FlowHandle targetHandle,
                                                              FlowHandle parentHandle) {
        assert(targetHandle >= 0 && parentHandle >= 0 && targetHandle < m_flows.size() &&
               parentHandle < m_flows.size());
        auto &targetBranch = m_flows[targetHandle];
        auto &parentBranch = m_flows[parentHandle];
        assert(!targetBranch.m_recycled);
        assert(!parentBranch.m_recycled);
        targetBranch.m_parentHandle = parentHandle;
        parentBranch.m_childHandles.emplace_back(targetHandle);
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    FlowHandle Skeleton<SkeletonData, FlowData, NodeData>::AllocateFlow() {
        if (m_flowPool.empty()) {
            auto newBranch = m_flows.emplace_back(m_flows.size());
            return newBranch.m_handle;
        }
        auto handle = m_flowPool.front();
        m_flowPool.pop();
        auto &flow = m_flows[handle];
        flow.m_recycled = false;
        return handle;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::RecycleFlowSingle(FlowHandle handle, const std::function<void(FlowHandle)>& flowHandler) {
        assert(!m_flows[handle].m_recycled);
        auto &flow = m_flows[handle];
        flowHandler(handle);
        flow.m_parentHandle = -1;
        flow.m_childHandles.clear();
        flow.m_nodes.clear();

        flow.m_data = {};
        flow.m_info = {};

        flow.m_recycled = true;
        flow.m_apical = false;
        m_flowPool.emplace(handle);
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::RecycleNodeSingle(NodeHandle handle, const std::function<void(NodeHandle)>& nodeHandler) {
        assert(!m_nodes[handle].m_recycled);
        auto &node = m_nodes[handle];
        nodeHandler(handle);
        node.m_parentHandle = -1;
        node.m_flowHandle = -1;
        node.m_endNode = true;
        node.m_childHandles.clear();

        node.m_data = {};
        node.m_info = {};

        node.m_recycled = true;
        m_nodePool.emplace(handle);
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    NodeHandle Skeleton<SkeletonData, FlowData, NodeData>::AllocateNode() {
        if (m_nodePool.empty()) {
            return m_nodes.emplace_back(m_nodes.size()).m_handle;
        }
        auto handle = m_nodePool.front();
        m_nodePool.pop();
        auto &node = m_nodes[handle];
        node.m_recycled = false;
        return handle;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    int Skeleton<SkeletonData, FlowData, NodeData>::GetVersion() const {
        return m_version;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::CalculateFlows() {
        for (const auto &flowHandle: m_sortedFlowList) {
            auto &flow = m_flows[flowHandle];
            auto &firstNode = m_nodes[flow.m_nodes.front()];
            auto &lastNode = m_nodes[flow.m_nodes.back()];
            flow.m_info.m_startThickness = firstNode.m_info.m_thickness;
            flow.m_info.m_globalStartPosition = firstNode.m_info.m_globalPosition;
            flow.m_info.m_globalStartRotation = firstNode.m_info.m_globalRotation;

            flow.m_info.m_endThickness = lastNode.m_info.m_thickness;
            flow.m_info.m_globalEndPosition = lastNode.m_info.m_globalPosition +
                                              lastNode.m_info.m_length *
                                              (lastNode.m_info.m_globalRotation * glm::vec3(0, 0, -1));
            flow.m_info.m_globalEndRotation = lastNode.m_info.m_globalRotation;
        }
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    void Skeleton<SkeletonData, FlowData, NodeData>::CalculateTransforms()
    {
        m_min = glm::vec3(FLT_MAX);
        m_max = glm::vec3(FLT_MIN);
        for (const auto& nodeHandle : m_sortedNodeList) {
            auto& node = m_nodes[nodeHandle];
            auto& nodeInfo = node.m_info;
            if (node.m_parentHandle != -1) {
                auto& parentInfo = m_nodes[node.m_parentHandle].m_info;
                nodeInfo.m_globalRotation =
                    parentInfo.m_globalRotation * nodeInfo.m_localRotation;
                nodeInfo.m_globalPosition =
                    parentInfo.m_globalPosition + parentInfo.m_length *
                    (parentInfo.m_globalRotation * glm::vec3(0, 0, -1))
                    + nodeInfo.m_localPosition;
                auto front = nodeInfo.m_globalRotation * glm::vec3(0, 0, -1);
                auto parentRegulatedUp = parentInfo.m_regulatedGlobalRotation * glm::vec3(0, 1, 0);
                auto regulatedUp = glm::normalize(glm::cross(glm::cross(front, parentRegulatedUp), front));
                nodeInfo.m_regulatedGlobalRotation = glm::quatLookAt(front, regulatedUp);
            }
            m_min = glm::min(m_min, nodeInfo.m_globalPosition);
            m_max = glm::max(m_max, nodeInfo.m_globalPosition);
            const auto endPosition = nodeInfo.m_globalPosition + nodeInfo.m_length *
                (nodeInfo.m_globalRotation *
                    glm::vec3(0, 0, -1));
            m_min = glm::min(m_min, endPosition);
            m_max = glm::max(m_max, endPosition);
        }
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    const std::vector<Flow<FlowData>> &Skeleton<SkeletonData, FlowData, NodeData>::RefRawFlows() const {
        return m_flows;
    }

    template<typename SkeletonData, typename FlowData, typename NodeData>
    const std::vector<Node<NodeData>> &
    Skeleton<SkeletonData, FlowData, NodeData>::RefRawNodes() const {
        return m_nodes;
    }

#pragma endregion
#pragma endregion
}