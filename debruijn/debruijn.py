#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
import textwrap
from operator import itemgetter
import random
from random import randint
import statistics
import networkx as nx
import matplotlib
random.seed(9001)
import matplotlib.pyplot as plt

matplotlib.use("Agg")

__author__ = "GBABOUA Cassandra"
__copyright__ = "Universite Paris Cite"
__credits__ = ["GBABOUA Cassandra"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "GBABOUA Cassandra"
__email__ = "cassandra.gbaboua@outlook.fr"
__status__ = "Developpement"


def isfile(path):  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file doesn't exist

    :return: (str) Path
    """
    if not os.path.isfile(path):
        if os.path.isdir(path):
            msg = "{0} is a directory".format(path)
        else:
            msg = "{0} does not exist.".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=str,
        default=os.curdir + os.sep + "contigs.fasta",
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=str, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file):
    """Extract reads from fastq files.

    :param fastq_file: (str) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r") as fastq:
        for line in fastq:
            yield (next(fastq).strip())
            next(fastq)
            next(fastq)


def cut_kmer(read, kmer_size):
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that iterate the kmers of of size kmer_size.
    """
    for i in range(0, len(read) - kmer_size + 1):
        yield read[i : i + kmer_size]


def build_kmer_dict(fastq_file, kmer_size):
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}  # Utilise un dictionnaire standard
    for read in read_fastq(fastq_file):  # Lit les séquences depuis le fichier FASTQ
        for kmer in cut_kmer(read, kmer_size):  # Découpe chaque séquence en k-mers
            if kmer in kmer_dict:  # Vérifie si le k-mer est déjà dans le dictionnaire
                kmer_dict[kmer] += 1  # Incrémente le compteur pour ce k-mer
            else:
                kmer_dict[
                    kmer
                ] = 1  # Ajoute le k-mer au dictionnaire avec une valeur de 1
    return kmer_dict


def build_graph(kmer_dict):
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    my_graph = nx.DiGraph()  # Crée un graphe orienté
    for kmer, weight in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]
        if my_graph.has_edge(prefix, suffix):
            my_graph[prefix][suffix]["weight"] += weight
        else:
            my_graph.add_edge(prefix, suffix, weight=weight)
    return my_graph


def remove_paths(graph, path_list, delete_entry_node, delete_sink_node):
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        nodes_to_remove = list(path[1:-1])
        if delete_entry_node:
            nodes_to_remove = [path[0]] + nodes_to_remove
        if delete_sink_node:
            nodes_to_remove = nodes_to_remove + [path[-1]]
        graph.remove_nodes_from(nodes_to_remove)
    return graph


def select_best_path(
    graph,
    path_list,
    path_length,
    weight_avg_list,
    delete_entry_node=False,
    delete_sink_node=False,
):
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # Calcul de l'écart-type pour weight_avg_list
    mean_weight = sum(weight_avg_list) / len(weight_avg_list)
    std_dev_weight = (
        sum((x - mean_weight) ** 2 for x in weight_avg_list) / len(weight_avg_list)
    ) ** 0.5

    best_path = None

    if std_dev_weight > 0:
        best_path = path_list[weight_avg_list.index(max(weight_avg_list))]
    else:
        # Calcul de l'écart-type pour path_length
        mean_length = sum(path_length) / len(path_length)
        std_dev_length = (
            sum((x - mean_length) ** 2 for x in path_length) / len(path_length)
        ) ** 0.5

        if std_dev_length > 0:
            best_path = path_list[path_length.index(max(path_length))]
        else:
            best_path = path_list[randint(0, len(path_list) - 1)]

    for path in path_list:
        if path != best_path:
            nodes_to_remove = path[1:-1]  # Noeuds intermédiaires
            if delete_entry_node:
                nodes_to_remove = [path[0]] + nodes_to_remove
            if delete_sink_node:
                nodes_to_remove = nodes_to_remove + [path[-1]]
            graph.remove_nodes_from(nodes_to_remove)

    return graph


def path_average_weight(graph, path):
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph, ancestor_node, descendant_node):
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    all_paths = list(
        nx.all_simple_paths(graph, source=ancestor_node, target=descendant_node)
    )
    path_length = [len(path) for path in all_paths]
    weight_avg_list = []

    for path in all_paths:
        weight_sum = 0
        for i in range(len(path) - 1):
            weight_sum += graph[path[i]][path[i + 1]]["weight"]
        weight_avg_list.append(weight_sum / len(path))

    graph = select_best_path(graph, all_paths, path_length, weight_avg_list)
    return graph


def simplify_bubbles(graph):
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    modified = True
    while modified:
        modified = False
        for node in list(graph.nodes()):
            predecessors = list(graph.predecessors(node))
            if len(predecessors) > 1:
                for i in range(len(predecessors)):
                    for j in range(i + 1, len(predecessors)):
                        node_i = predecessors[i]
                        node_j = predecessors[j]
                        lowest_common_ancestor = nx.lowest_common_ancestor(
                            graph, node_i, node_j
                        )

                        if lowest_common_ancestor is not None:
                            modified = True
                            graph = solve_bubble(graph, lowest_common_ancestor, node)
                            break
                    if modified:
                        break
            if modified:
                break
    return graph


def solve_entry_tips(graph, starting_nodes):
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph:
        node_pred = list(graph.predecessors(node))
        if len(node_pred) > 1:
            paths = [
                list(nx.all_simple_paths(graph, node_start_i, node))
                for node_start_i in starting_nodes
            ]
            paths = [path[0] for path in paths if len(path) > 0]
            lengths = [len(path) - 1 for path in paths]
            weights = [
                path_average_weight(graph, path)
                if lengths[i] > 1
                else graph.get_edge_data(*path)["weight"]
                for i, path in enumerate(paths)
            ]

            graph = select_best_path(
                graph,
                paths,
                lengths,
                weights,
                delete_entry_node=True,
                delete_sink_node=False,
            )
            graph = solve_entry_tips(graph, starting_nodes)
            break

    return graph


def solve_out_tips(graph, ending_nodes):
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    for node in graph:
        node_success = list(graph.successors(node))
        if len(node_success) > 1:
            paths = [
                list(nx.all_simple_paths(graph, node, node_end_i))
                for node_end_i in ending_nodes
            ]
            paths = [path[0] for path in paths if len(path) > 0]
            lengths = [len(path) - 1 for path in paths]
            weights = [
                path_average_weight(graph, path)
                if lengths[i] > 1
                else graph.get_edge_data(*path)["weight"]
                for i, path in enumerate(paths)
            ]

            graph = select_best_path(
                graph,
                paths,
                lengths,
                weights,
                delete_entry_node=False,
                delete_sink_node=True,
            )
            graph = solve_out_tips(graph, ending_nodes)
            break

    return graph


def get_starting_nodes(graph):
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = []
    for node, degree in graph.in_degree():
        if degree == 0:
            starting_nodes.append(node)
    return starting_nodes


def get_sink_nodes(graph):
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = []
    for node, degree in graph.out_degree():
        if degree == 0:
            sink_nodes.append(node)
    return sink_nodes


def get_contigs(graph, starting_nodes, ending_nodes):
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            if nx.has_path(graph, start, end):
                for path in nx.all_simple_paths(graph, start, end):
                    contig = path[0]
                    for node in path[1:]:
                        contig += node[-1]
                    contigs.append((contig, len(contig)))
    return contigs


def save_contigs(contigs_list, output_file):
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (str) Path to the output file
    """
    with open(output_file, "w") as file:
        for i, (contig, length) in enumerate(contigs_list):
            file.write(">contig_" + str(i) + " len=" + str(length) + "\n")
            formatted_contig = textwrap.fill(contig, width=80)
            file.write(formatted_contig + "\n")


def draw_graph(graph, graphimg_file):  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (str) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file)


# ==============================================================
# Main program
# ==============================================================
def main():  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    fastq_file = args.fastq_file  # Mettez le chemin vers votre fichier FASTQ ici
    kmer_size = args.kmer_size
    kmer_dict = build_kmer_dict(fastq_file, kmer_size)
    debruijn_graph = build_graph(kmer_dict)
    starting_nodes = get_starting_nodes(debruijn_graph)
    sink_nodes = get_sink_nodes(debruijn_graph)
    contigs = get_contigs(debruijn_graph, starting_nodes, sink_nodes)
    output_file = args.output_file
    save_contigs(contigs, output_file)
    # Voir le dictionnaire construit
    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    if args.graphimg_file:
        draw_graph(debruijn_graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
