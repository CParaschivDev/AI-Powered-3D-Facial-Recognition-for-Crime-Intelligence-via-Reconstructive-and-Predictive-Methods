"""
Generates a high-level Security & Governance stack diagram.
Requires: graphviz Python package and Graphviz system binaries.

Usage:
    pip install graphviz
    python security_governance_stack.py --output security_stack.svg

This script produces an SVG (default) or PNG file illustrating:
- Left: Users/Roles
- Middle: FastAPI + RBAC (AuthN, AuthZ, AuditLog)
- Right: Protected resources (Evidence store, PostgreSQL, Logs+Monitoring)
Annotated arrows show protections (TLS+JWT, AES-256 at rest, Redacted JSON logs).
"""
import argparse
from graphviz import Digraph


def build_graph():
    g = Digraph('G', format='svg')
    g.attr(rankdir='LR', fontsize='12')
    g.attr('node', shape='rect', style='filled', fillcolor='#f5f5f5', fontname='Helvetica')

    # Left: Users / Roles
    with g.subgraph(name='cluster_users') as c:
        c.attr(label='Users & Roles', color='#2b7cff')
        c.node('Admin', fillcolor='#dbeeff')
        c.node('Analyst', fillcolor='#dbeeff')
        c.node('Operator', fillcolor='#dbeeff')
        c.node('Auditor', fillcolor='#dbeeff')

    # Middle: API / FastAPI + RBAC
    with g.subgraph(name='cluster_api') as a:
        a.attr(label='FastAPI + RBAC', color='#2ecc71')
        a.node('API', shape='component', fillcolor='#e8f8f0')
        # internal components as separate nodes
        a.node('AuthN\n(IdP / OAuth2 / OIDC)')
        a.node('AuthZ\n(RBAC policy)')
        a.node('AuditLog\n(writer)')
        # layout edges inside cluster
        a.edge('API', 'AuthN\n(IdP / OAuth2 / OIDC)', style='dashed', color='#2ecc71')
        a.edge('API', 'AuthZ\n(RBAC policy)', style='dashed', color='#2ecc71')
        a.edge('API', 'AuditLog\n(writer)', style='dotted', color='#7f8c8d')

    # Right: Protected resources
    with g.subgraph(name='cluster_resources') as r:
        r.attr(label='Protected resources', color='#e67e22')
        r.node('Evidence\n(Encrypted object store)', fillcolor='#fff3e0')
        r.node('PostgreSQL\n(Evidence, Snapshots, Predictions, ModelVersion, AuditLog, BiasMetrics)', fillcolor='#fff3e0')
        r.node('Logs + Monitoring\n(ELK/Opensearch, Prometheus, Jaeger)', fillcolor='#fff3e0')

    # Edges between User and API (annotated with protections)
    g.edge('Admin', 'API', label='TLS + JWT', color='#34495e')
    g.edge('Analyst', 'API', label='TLS + JWT', color='#34495e')
    g.edge('Operator', 'API', label='TLS + JWT', color='#34495e')
    g.edge('Auditor', 'API', label='TLS + JWT', color='#34495e')

    # API -> Resources with annotations
    g.edge('API', 'Evidence\n(Encrypted object store)', label='AES-256 at rest / envelope encryption', color='#8e44ad')
    g.edge('API', 'PostgreSQL\n(Evidence, Snapshots, Predictions, ModelVersion, AuditLog, BiasMetrics)', label='AES-256 at rest / envelope encryption', color='#8e44ad')
    g.edge('API', 'Logs + Monitoring\n(ELK/Opensearch, Prometheus, Jaeger)', label='Redacted JSON logs', color='#c0392b')

    # AuditLog flow to Logs+Monitoring
    g.edge('AuditLog\n(writer)', 'Logs + Monitoring\n(ELK/Opensearch, Prometheus, Jaeger)', label='Redacted audit records', style='dashed', color='#7f8c8d')

    return g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render security & governance stack diagram')
    parser.add_argument('--output', '-o', default='security_stack.svg', help='Output filename (svg or png)')
    args = parser.parse_args()

    graph = build_graph()
    out_path = graph.render(filename=args.output, cleanup=True)
    print(f'Wrote {out_path}')
