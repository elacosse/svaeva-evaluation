# type: ignore[attr-defined]
import asyncio
import os
import re

import dotenv
import numpy as np
import typer
from pyvis.network import Network
from rich.console import Console

from svaeva_evaluation import version
from svaeva_evaluation.audio.generate import async_generate_audio_from_list
from svaeva_evaluation.conversation import retrieve_redis_windowed_chat_history_as_text
from svaeva_evaluation.network import (
    calculate_distances_between_users,
    generate_network_from_edges,
    select_best_connected_node,
    threshold_distances,
)
from svaeva_evaluation.visualization.plotting import (
    plot_3d_pca_embeddings,
    plot_edge_distribution,
    plot_tSNE_embeddings,
    return_circle_cropped_image_from_user,
)

app = typer.Typer(
    name="svaeva-evaluation",
    help="`svaeva-evaluation` is a Python cli/package",
    add_completion=False,
)
console = Console()


def version_callback(print_version: bool) -> None:
    """Print the version of the package."""
    if print_version:
        console.print(f"[yellow]svaeva-evaluation[/] version: [bold blue]{version}[/]")
        raise typer.Exit()


def extract_conversation_from_user(
    user_id: str,
) -> str:
    """Extract the conversation from a user object."""

    from svaeva_redux.schemas.redis import UserModel

    try:
        user = UserModel.get(user_id)
    except Exception as e:
        console.log(f"An error occurred: {e}")
        return ""

    session_id = user_id
    url = os.getenv("REDIS_OM_URL")
    key_prefix = f"{os.getenv('PLATFORM_ID')}_{os.getenv('CONVERSATION_ID')}:"
    chat_history_length = 30
    conversation = retrieve_redis_windowed_chat_history_as_text(session_id, url, key_prefix, chat_history_length)
    return conversation


def patternize_list(input_list):
    pattern = re.compile(r"[\d\.]+")
    filtered_list = [pattern.sub("", word) for word in input_list]
    pattern_list = [" ... ".join([word] * 3) for word in filtered_list]
    return pattern_list


def construct_network(edges: list, edge_weights: list, root_path: str):
    for edge in edges:  # normalize edge weights
        edge["distance"] = (edge["distance"] - min(edge_weights)) / (max(edge_weights) - min(edge_weights))

    network = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    nx_graph = generate_network_from_edges(edges)
    network.from_nx(nx_graph)
    save_path = os.path.join(root_path, "data/network", "network.html")
    network.save_graph(save_path)
    console.log(f"Saved network graph to: {save_path}")


def select_node_from_network(edges: list):
    best_node = select_best_connected_node(edges)
    console.log(f"Best connected node: [green]{best_node}[/]")
    return best_node


async def generate_patternized_audio(conversation_text: str):
    from svaeva_evaluation.conversation import (
        construct_word_narrative_with_hurt,
        construct_word_narrative_without_hurt,
    )

    word_narrative_positive = construct_word_narrative_without_hurt(conversation_text)
    console.log(f"[green]Joy narrative: {word_narrative_positive} [/]")
    word_narrative_negative = construct_word_narrative_with_hurt(conversation_text)
    console.log(f"[red]Hurt narrative: {word_narrative_negative} [/]")
    list_text = patternize_list(word_narrative_positive)
    await async_generate_audio_from_list(list_text, "Emily", "positive")
    list_text = patternize_list(word_narrative_negative)
    await async_generate_audio_from_list(list_text, "Emily", "negative")


@app.command(name="")
def main(
    command=typer.Argument(..., help="Available commands: 'plot' ."),
    print_version: bool = typer.Option(
        None,
        "-v",
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Prints the version of the svaeva-evaluation package.",
    ),
) -> None:
    dotenv.load_dotenv()
    platform_id = os.getenv("PLATFORM_ID")
    group_id = os.getenv("GROUP_ID")
    conversation_id = os.getenv("CONVERSATION_ID")
    interaction_count = int(os.getenv("INTERACTION_COUNT", 2))
    key_prefix = f"{platform_id}_{conversation_id}:"

    console.log(f"Command: {command}")

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    console.log(f"Root path: {root_path}")

    console.log("Gathering Users from Redis...")
    console.log("[red]Redis Host[/]: " + os.environ["REDIS_HOST"])
    console.log("[red]Redis OM URL[/]: " + os.environ["REDIS_OM_URL"])
    console.log(f"Group ID: {group_id}")
    console.log(f"Key Prefix: {key_prefix}")
    console.log(f"Interaction Count Filter: {interaction_count}")

    from svaeva_redux.schemas.redis import UserModel

    users = UserModel.find(
        (UserModel.group_id == group_id)
        & (UserModel.platform_id == platform_id)
        & (UserModel.interaction_count >= interaction_count)
    ).all()
    console.log(f"Number of users: {len(users)}")

    if command == "save-avatar-images":
        console.log("Saving avatars...")
        for user in users:
            return_circle_cropped_image_from_user(user)

    console.log("Calculating distances between user conversation embeddings...")
    distances = calculate_distances_between_users(users)

    # select the lowest 10% of edge weights as threshold
    edge_weights = []
    for distance in distances:
        for key, value in distance["distance"].items():
            edge_weights.append(value)
    threshold = np.quantile(edge_weights, 0.1)
    console.log("Selected threshold for lowest quantile: ", threshold)

    # filter out edges with weights above the threshold
    console.log("Edges before thresholding: ", len(distances) * len(distances[0]["distance"]))
    edges = threshold_distances(distances, threshold)
    console.log("Edges after thresholding: ", len(edges))  # n x n - n expected from simulation
    console.log("Percentage of edges kept: ", len(edges) / (len(distances) * len(distances[0]["distance"])) * 100)

    if command == "network":
        construct_network(edges, edge_weights, root_path)

    if command == "select":
        """Select the best connected node and save the conversation to a file."""
        best_node = select_node_from_network(edges)
        conversation_from_best_node = extract_conversation_from_user(best_node)
        # Save the conversation to a file
        save_path = os.path.join(root_path, "data/conversations", "best_node_conversation.txt")
        with open(save_path, "w") as f:
            f.write(conversation_from_best_node)
        console.log(f"Saved conversation from best node to: {save_path}")

    if command == "audio":
        """Generate audio from the best connected node."""
        console.log("Generating audio from word narrative...")
        best_node = select_node_from_network(edges)
        conversation_text = extract_conversation_from_user(best_node)
        asyncio.run(generate_patternized_audio(conversation_text))

    if command == "plot":
        """Plot tSNE embeddings and edge distribution."""
        console.log("Plotting tSNE embedding clustering...")
        embeddings = [np.array(user.conversation_embedding).reshape(1, -1) for user in users]
        names = []
        for user in users:
            parts = user.id.split("-")
            name = parts[2]
            names.append(name)

        fig = plot_tSNE_embeddings(embeddings, 2, names, 15)
        save_path = os.path.join(root_path, "data/plots", "tSNE_embeddings.png")
        fig.savefig(save_path)
        console.log(f"Saved tSNE plot to: {save_path}")

        fig = plot_3d_pca_embeddings(users, names)
        save_path = os.path.join(root_path, "data/plots", "PCA_embeddings.png")
        fig.savefig(save_path)
        console.log(f"Saved 3d-PCA plot to: {save_path}")

        fig = plot_edge_distribution(distances)
        save_path = os.path.join(root_path, "data/plots", "edge_distribution.png")
        fig.savefig(save_path)
        console.log(f"Saved edge distribution plot to: {save_path}")

    console.log("Done!")


if __name__ == "__main__":
    app()
