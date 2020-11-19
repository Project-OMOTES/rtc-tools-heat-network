from rtctools_heat_network.pycml import Model as _Model


RETRY_LOOP_LIMIT = 100


class _RetryLaterException(Exception):
    pass


class _ESDLModelBase(_Model):
    def _esdl_convert(self, converter, assets, prefix):

        # Sometimes we need information of one component in order to convert
        # another. For example, the nominal discharg of a pipe is used to set
        # the nominal discharge of its connected components.
        retry_assets = list(assets.values())

        for _ in range(RETRY_LOOP_LIMIT):

            current_assets = retry_assets
            retry_assets = []

            for asset in current_assets:
                try:
                    pycml_type, modifiers = converter.convert(asset)
                except _RetryLaterException:
                    retry_assets.append(asset)
                    continue

                self.add_variable(pycml_type, asset.name, **modifiers)

            if not retry_assets:
                break
        else:
            raise Exception("Parsing of assets exceeded maximum iteration limit.")

        in_suf = f"{prefix}In"
        out_suf = f"{prefix}Out"
        node_suf = f"{prefix}Conn"

        node_assets = [a for a in assets.values() if a.asset_type == "Joint"]
        non_node_assets = [a for a in assets.values() if a.asset_type != "Joint"]

        # First we map all port ids to their respective PyCML ports. We only
        # do this for non-nodes, as for nodes we don't quite know what port
        # index a connection has to use yet.
        port_map = {}

        for asset in non_node_assets:
            component = getattr(self, asset.name)

            port_map[asset.in_port.id] = getattr(component, in_suf)
            port_map[asset.out_port.id] = getattr(component, out_suf)

        # Nodes are special in that their in/out ports can have multiple
        # connections. This means we have some bookkeeping to do per node. We
        # therefore do the nodes first, and do all remaining connections
        # after.
        connections = set()

        for asset in node_assets:
            component = getattr(self, asset.name)

            i = 1
            for port in (asset.in_port, asset.out_port):
                for connected_to in port.connectedTo.items:
                    conn = (port.id, connected_to.id)
                    if conn in connections or tuple(reversed(conn)) in connections:
                        continue

                    self.connect(getattr(component, node_suf)[i], port_map[connected_to.id])
                    connections.add(conn)
                    i += 1

        # All non-Joints/nodes
        for asset in non_node_assets:
            for port in (asset.in_port, asset.out_port):
                for connected_to in port.connectedTo.items:
                    conn = (port.id, connected_to.id)
                    if conn in connections or tuple(reversed(conn)) in connections:
                        continue

                    self.connect(port_map[port.id], port_map[connected_to.id])
                    connections.add(conn)
