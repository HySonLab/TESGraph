def __getitem__(self, index):
        #try:
            timers = dict()
            timer = TicToc()
            scan_id = snp.unpack(self.scans, index)  # self.scans[idx]

            '''open data'''
            timer.tic()
            # open
            self.open_filtered()
            self.open_data()

            # get SG data
            scan_data_raw = self.sg_data[scan_id]
            scan_data = raw_to_data(scan_data_raw)
            # shortcut
            object_data = scan_data['nodes']
            relationships_data = scan_data['relationships']

            filtered_data = raw_to_data(self.filtered_data[scan_id])
            filtered_node_indices = filtered_data[define.NAME_FILTERED_OBJ_INDICES]
            filtered_kf_indices = filtered_data[define.NAME_FILTERED_KF_INDICES]

            mv_data = None
            if self.mconfig.load_images:
                self.open_mv_graph()

                mv_data = self.mv_data[scan_id]
                mv_nodes = mv_data['nodes']  # contain kf ids of a given node
                if self.mconfig.is_roi_img:
                    self.open_img()
                    roi_imgs = self.roi_imgs[scan_id]

            '''filter node data'''
            object_data = {nid: object_data[nid] for nid in filtered_node_indices}

            timers['open_data'] = timer.tocvalue()

            ''' build nn dict '''
            timer.tic()
            nns = dict()
            seg2inst = dict()
            for oid, odata in object_data.items():
                nns[str(oid)] = [int(s) for s in odata['neighbors']]

                '''build instance dict'''
                if 'instance_id' in odata:
                    seg2inst[oid] = odata['instance_id']
                else:
                    seg2inst[oid]
            timers['build_nn_dict'] = timer.tocvalue()

            ''' load point cloud data '''
            timer.tic()
            if self.mconfig.load_points:
                if 'scene' in scan_id:
                    path = os.path.join(self.root_scannet, scan_id)
                else:
                    path = os.path.join(self.root_3rscan, scan_id)

                if self.config.data.load_cache:
                    data = self.cache_data[scan_id]
                else:
                    data = load_mesh(path, self.label_file,
                                    self.use_rgb, self.use_normal)
                points = copy.deepcopy(data['points'])
                instances = copy.deepcopy(data['instances'])

                if self.use_data_augmentation and not self.for_eval:
                    points = self.data_augmentation(points)
            timers['load_pc'] = timer.tocvalue()

            '''extract 3D node classes and instances'''
            timer.tic()
            cat, oid2idx, idx2oid, filtered_instances = self.__sample_3D_nodes(object_data,
                                                                            mv_data,
                                                                            nns)
            timers['sample_3D_nodes'] = timer.tocvalue()

            '''sample 3D node connections'''
            timer.tic()
            edge_indices_3D = self.__sample_3D_node_edges(
                cat, oid2idx, filtered_instances, nns)
            timers['sample_3D_node_edges'] = timer.tocvalue()

            '''extract relationships data'''
            timer.tic()
            relationships_3D = self.__extract_relationship_data(
                relationships_data, oid2idx)
            timers['extract_relationship_data'] = timer.tocvalue()

            # relationships_3D_mask = [] # change obj idx to obj mask idx

            ''' 
            Generate mapping from selected entity buffer to the ground truth entity buffer (for evaluation)
            Save the mapping in edge_index format to allow PYG to rearrange them.
            '''
            instance2labelName = {int(key): node['label']
                                for key, node in object_data.items()}
            # Collect GT entity list
            gt_entities = set()
            gtIdx_entities_cls = []
            gtIdx2ebIdx = []
            for key, value in relationships_3D.items():
                sub_o = key[0]
                tgt_o = key[1]
                gt_entities.add(sub_o)
                gt_entities.add(tgt_o)
            gt_entities = [k for k in gt_entities]
            # assert len(gt_entities) > 0
            for gtIdx, k in enumerate(gt_entities):
                if k in oid2idx:
                    idx = oid2idx[k]
                    gtIdx2ebIdx.append([gtIdx, idx])
                    label = instance2labelName[k]
                    gtIdx_entities_cls.append(self.classNames.index(label))
                else:
                    # Add negative index to indicate missing
                    gtIdx2ebIdx.append([gtIdx, -1])

            gtIdx_edge_index = []
            gtIdx_edge_cls = []
            for key, value in relationships_3D.items():
                sub_o = key[0]
                tgt_o = key[1]
                # sub_cls = instance2labelName[sub_o]
                # tgt_cls = instance2labelName[tgt_o]
                # sub_cls_id = self.classNames.index(sub_cls)
                # tgt_cls_id = self.classNames.index(tgt_cls)
                # relationships_3D_mask.append([sub_o,tgt_o,sub_cls_id,tgt_cls_id,value])

                sub_ebIdx = oid2idx[sub_o]
                tgt_ebIdx = oid2idx[tgt_o]
                sub_gtIdx = gt_entities.index(sub_o)
                tgt_gtIdx = gt_entities.index(tgt_o)
                gtIdx_edge_index.append([sub_gtIdx, tgt_gtIdx])
                gtIdx_edge_cls.append(value)

            # gtIdx_entities_cls = torch.from_numpy(np.array(gtIdx_entities_cls))
            gtIdx2ebIdx = torch.tensor(
                gtIdx2ebIdx, dtype=torch.long).t().contiguous()
            # gtIdx_edge_cls = torch.from_numpy(np.array(gtIdx_edge_cls))
            gtIdx_edge_index = torch.tensor(
                gtIdx_edge_index, dtype=torch.long).t().contiguous()

            '''sample 3D edges'''
            timer.tic()
            gt_rels_3D, edge_index_has_gt_3D = self.__sample_relationships(
                relationships_3D, idx2oid, edge_indices_3D)
            timers['sample_relationships'] = timer.tocvalue()

            '''drop edges'''  # to fit memory
            gt_rels_3D, edge_indices_3D = self.__drop_edge(
                gt_rels_3D, edge_indices_3D, edge_index_has_gt_3D)

            ''' random sample points '''
            if self.mconfig.load_points:
                timer.tic()
                obj_points, descriptor, bboxes = self.__sample_points(
                    scan_id, points, instances, cat, filtered_instances)
                timers['sample_points'] = timer.tocvalue()

                bboxes_tensor = torch.Tensor(np.array(bboxes))
                bboxes_tensor = torch.reshape(bboxes_tensor, 
                                            (bboxes_tensor.shape[0], bboxes_tensor.shape[1]*bboxes_tensor.shape[2]))

                '''build rel points'''
                timer.tic()
                if self.mconfig.rel_data_type == 'points':
                    rel_points = self.__sample_rel_points(
                        points, instances, idx2oid, bboxes, edge_indices_3D)
                timers['sample_rel_points'] = timer.tocvalue()

            '''load images'''
            if self.mconfig.load_images:
                timer.tic()
                if self.mconfig.is_roi_img:
                    roi_images, node_descriptor_for_image, edge_indices_img_to_obj = \
                        self.__load_roi_images(cat, idx2oid, mv_nodes, roi_imgs,
                                            object_data, filtered_instances)
                else:
                    images, img_bounding_boxes, bbox_cat, node_descriptor_for_image, \
                        image_edge_indices, img_idx2oid, temporal_node_graph, temporal_edge_graph = self.__load_full_images(
                            scan_id, idx2oid, cat, scan_data, mv_data, filtered_kf_indices)
                    relationships_img = self.__extract_relationship_data(
                        relationships_data, oid2idx)
                    gt_rels_2D, edge_index_has_gt_2D = self.__sample_relationships(
                        relationships_img, img_idx2oid, image_edge_indices)

                    # img_oid_indices = [oid for oid in img_idx2oid.values()]
                    img_oid_indices = [seg2inst[oid]
                                    for oid in img_idx2oid.values()]
                    img_oid_indices = torch.from_numpy(np.array(img_oid_indices))
                    # gt_rels_2D, image_edge_indices, final_edge_indices_2D = self.__drop_edge(
                    #     gt_rels_2D, image_edge_indices,edge_index_has_gt_2D)
                    # # filter temporal edge graph
                    # to_delete=[]
                    # all_indices = range(len(temporal_edge_graph))
                    # for idx in all_indices:
                    #     idx_0,idx_1 = temporal_edge_graph[idx][0],temporal_edge_graph[idx][1]
                    #     if idx_0 not in final_edge_indices_2D or idx_1 not in final_edge_indices_2D:
                    #         to_delete.append(idx)
                    # to_keep = set(all_indices).difference(to_delete)
                    # temporal_edge_graph = [temporal_edge_graph[idx] for idx in to_keep]

                    '''to tensor'''
                    assert len(img_bounding_boxes) > 0
                    images = torch.stack(images, dim=0)
                    assert len(bbox_cat) == len(img_bounding_boxes)
                    img_bounding_boxes = torch.from_numpy(
                        np.array(img_bounding_boxes)).float()
                    gt_class_image = torch.from_numpy(np.array(bbox_cat))
                    image_edge_indices = torch.tensor(
                        image_edge_indices, dtype=torch.long)
                    temporal_node_graph = torch.tensor(
                        temporal_node_graph, dtype=torch.long)
                    temporal_edge_graph = torch.tensor(
                        temporal_edge_graph, dtype=torch.long)
                    if len(node_descriptor_for_image) > 0:
                        node_descriptor_for_image = torch.stack(
                            node_descriptor_for_image)
                    else:
                        node_descriptor_for_image = torch.tensor(
                            [], dtype=torch.long)
                timers['load_images'] = timer.tocvalue()

            '''collect attribute for nodes'''
            # for inseg the segment instance should be converted back to the GT instances
            inst_indices = [seg2inst[k] for k in idx2oid.values()]

            ''' to tensor '''
            gt_class_3D = torch.from_numpy(np.array(cat))
            tensor_oid = torch.from_numpy(np.array(inst_indices))

            edge_indices_3D = torch.tensor(edge_indices_3D, dtype=torch.long)
            # new_edge_index_has_gt = torch.tensor(new_edge_index_has_gt,dtype=torch.long)
            # idx2iid = seg2inst
            # idx2iid = torch.LongTensor([seg2inst[oid] if oid in seg2inst else oid for oid in idx2oid.values() ]) # mask idx to instance idx
            # idx2oid = torch.LongTensor([oid for oid in idx2oid.values()]) # mask idx to seg idx (instance idx)

            '''Gather output in HeteroData'''
            output = HeteroData()
            output['scan_id'] = scan_id  # str

            output['node'].x = torch.zeros([gt_class_3D.shape[0], 1])  # dummy
            output['node'].y = gt_class_3D
            output['node'].oid = tensor_oid

            # if len(gtIdx_entities_cls) == 0:
            #     print('scan_id',scan_id)
            #     print('len(gtIdx_entities_cls)',len(gtIdx_entities_cls))
            #     print('gtIdx2ebIdx.shape',gtIdx2ebIdx.shape)
            #     print('gtIdx_edge_cls',len(gtIdx_edge_cls))
            #     print('gtIdx_edge_index.shape',gtIdx_edge_index.shape)
            #     print('hallo')
            # gtIdx_entities_cls =None
            # gtIdx_edge_cls = None
            output['node_gt'].x = torch.zeros(
                [len(gtIdx_entities_cls), 1])  # dummy
            output['node_gt'].clsIdx = gtIdx_entities_cls if len(
                gtIdx_entities_cls) > 0 else torch.zeros([len(gtIdx_entities_cls), 1])  # dummy
            output['node_gt', 'to', 'node'].edge_index = gtIdx2ebIdx
            output['node_gt', 'to', 'node_gt'].clsIdx = gtIdx_edge_cls if len(
                gtIdx_edge_cls) > 0 else torch.zeros([len(gtIdx_entities_cls), 1])  # dummy
            output['node_gt', 'to', 'node_gt'].edge_index = gtIdx_edge_index

            # edges for computing features
            output['node', 'to', 'node'].edge_index = edge_indices_3D.t().contiguous()
            output['node', 'to', 'node'].y = gt_rels_3D

            if self.mconfig.load_points:
                output['node'].pts = obj_points

                if 'edge_desc' not in self.mconfig or self.mconfig['edge_desc'] == 'pts':
                    output['node'].desp = descriptor
                    output['coord'].x = bboxes_tensor

                if self.mconfig.rel_data_type == 'points':
                    output['edge'].pts = rel_points

            if self.mconfig.load_images:
                if self.mconfig.is_roi_img and type(roi_images):
                    output['roi'].x = torch.zeros([roi_images.size(0), 1])
                    output['roi'].img = roi_images
                    output['roi', 'sees', 'node'].edge_index = edge_indices_img_to_obj

                    if 'edge_desc' not in self.mconfig or self.mconfig['edge_desc'] == 'roi':
                        output['node'].desp = node_descriptor_for_image

                    # if not self.mconfig.load_points:
                    #     output['node'].desp = node_descriptor_for_image
                else:
                    output['roi'].x = torch.zeros([len(img_bounding_boxes), 1])
                    output['roi'].y = gt_class_image
                    output['roi'].box = img_bounding_boxes
                    output['roi'].img = images
                    output['roi'].desp = node_descriptor_for_image
                    output['roi'].oid = img_oid_indices

                    # need this for temporal edge graph
                    output['edge2D'].x = torch.zeros([len(temporal_node_graph), 1])

                    output['roi', 'to',
                        'roi'].edge_index = image_edge_indices.t().contiguous()
                    output['roi', 'to', 'roi'].y = gt_rels_2D
                    output['roi', 'temporal',
                        'roi'].edge_index = temporal_node_graph.t().contiguous()
                    output['edge2D', 'temporal',
                        'edge2D'].edge_index = temporal_edge_graph.t().contiguous()

                    # print('image_edge_indices',image_edge_indices)
                    # print('gt_rels_2D',gt_rels_2D)
                    # print(image_edge_indices)
                    # print(image_edge_indices)
                    # tmp1 = torch.sort(torch.unique(tensor_oid))[0]
                    # tmp2 = torch.sort(torch.unique(img_oid_indices))[0]
                    # tmp3 = torch.sort(torch.unique(torch.from_numpy(np.array([seg2inst[k] for k in tmp2.tolist()]))))[0]
                    # assert torch.equal(tmp1,tmp3)
                    # assert torch.equal(tmp1,tmp2)

            '''release'''
            self.reset_data()
            return output
        
        # except:
        #     print("\nDATA LOADER GETITEM ERROR. SKIP THIS DATA")
        #     return 0