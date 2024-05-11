# type: ignore[attr-defined]
import asyncio
import base64
import datetime
import json
import os
import re
from pathlib import Path
from typing import List

import dotenv
import numpy as np
import redis
import typer
from pyvis.network import Network
from rich.console import Console
from typing_extensions import Annotated

from svaeva_evaluation.audio.generate import async_generate_audio_from_list
from svaeva_evaluation.conversation import retrieve_redis_windowed_chat_history_as_text
from svaeva_evaluation.messaging import queue_message_to_user
from svaeva_evaluation.network import (
    calculate_distances_between_users,
    generate_network_from_edges,
    select_best_connected_nodes,
    threshold_distances,
)
from svaeva_evaluation.visualization.plotting import (
    plot_3d_pca_embeddings,
    plot_edge_distribution,
    plot_tSNE_embeddings,
    return_image_from_user,
)

dotenv.load_dotenv()
from svaeva_redux.schemas.redis import UserModel, UserVideoModel

DEFAULT_MESSAGE = "This is Consonância. You're invited to enter the room of healing algorithms for something special. Please type or click with /iamready if you accept this invitation. You have 10 minutes to accept this invitation."
platform_id = os.getenv("PLATFORM_ID")
group_id = os.getenv("GROUP_ID")
conversation_id = os.getenv("CONVERSATION_ID")
root_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
key_prefix = f"{platform_id}_{conversation_id}:"


app = typer.Typer(
    name="svaeva-evaluation",
    help="`svaeva-evaluation` is a Python cli/package to manage the ConsonâncIA installation",
    add_completion=False,
)
console = Console()
console.log(f"Root path: {root_path}")
console.log("[red]Redis Host[/]: " + os.environ["REDIS_HOST"])
console.log("[red]Redis OM URL[/]: " + os.environ["REDIS_OM_URL"])
console.log(f"[yellow]Group ID[/]: {group_id}")
console.log(f"[yellow]Key Prefix[/]: {key_prefix}")

# make sure the redis connection is working
try:
    conn = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB_INDEX"))
    conn.ping()
    console.log("[red]Redis[/] connection [green]successful![/]")
except Exception as e:
    console.log(f"An error occurred: {e}")
    raise typer.Exit()


def extract_conversation_from_user(
    user_id: str,
) -> str:
    """Extract the conversation from a user object."""

    from svaeva_redux.schemas.redis import UserModel

    try:
        _ = UserModel.get(user_id)
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


async def generate_patternized_audio(conversation_text: str, save_dir: Path):
    from svaeva_evaluation.conversation import (
        construct_word_narrative_with_hurt,
        construct_word_narrative_without_hurt,
    )

    word_narrative_positive = construct_word_narrative_without_hurt(conversation_text)
    console.log(f"[green]Joy narrative: {word_narrative_positive} [/]")
    word_narrative_negative = construct_word_narrative_with_hurt(conversation_text)
    console.log(f"[red]Hurt narrative: {word_narrative_negative} [/]")
    list_text = patternize_list(word_narrative_positive)
    await async_generate_audio_from_list(list_text, "Emily", "positive", save_dir)
    list_text = patternize_list(word_narrative_negative)
    await async_generate_audio_from_list(list_text, "Emily", "negative", save_dir)


def construct_and_save_network(edges: list, edge_weights: list) -> Network:
    for edge in edges:  # normalize edge weights
        edge["distance"] = (edge["distance"] - min(edge_weights)) / (max(edge_weights) - min(edge_weights))

    network = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    nx_graph = generate_network_from_edges(edges)
    network.from_nx(nx_graph)
    save_path = root_path / "data/network" / f"{group_id}-{platform_id}" / "network.html"
    # Create the directory if it doesn't exist
    if not os.path.exists(save_path.parent):
        os.makedirs(save_path.parent)
    network.save_graph(str(save_path))
    console.log(f"[green]Saved[/] network visualization to: {save_path}")

    # save network as a json
    network_json_string = str(network)
    data = json.loads(network_json_string)
    # add id key to each node
    modified_nodes = [{"id": node} for node in data["Nodes"]]
    data["Nodes"] = modified_nodes
    modified_json_string = json.dumps(data, indent=4)
    save_path = root_path / "data/network" / f"{group_id}-{platform_id}" / "network.json"
    with open(save_path, "w") as f:
        f.write(modified_json_string)
    console.log(f"[green]Saved[/] graph to: {save_path}")


def get_users(
    group_id: str,
    platform_id: str,
    interaction_count: int = -1,
    last_user_update_delta_seconds: float = -1,
    upper_bound_users: int = -1,
) -> List[UserModel]:
    users = UserModel.find(
        (UserModel.group_id == group_id)
        & (UserModel.platform_id == platform_id)
        & (UserModel.interaction_count >= interaction_count)
    ).all()
    if last_user_update_delta_seconds > 0:
        time_lower_bound = datetime.now().timestamp() - last_user_update_delta_seconds
        users = UserModel.find(
            (UserModel.group_id == group_id)
            & (UserModel.platform_id == platform_id)
            & (UserModel.interaction_count >= interaction_count)
            & (UserModel.date_updated_timestamp >= time_lower_bound)
        ).all()

    if upper_bound_users > 0:
        users = sorted(users, key=lambda x: x.date_updated_timestamp, reverse=True)
        users = users[:upper_bound_users]
        # display last user update timestamp (how long?)
        time_window = datetime.datetime.now().timestamp() - users[-1].date_updated_timestamp
        # convert to datetime
        time_window = datetime.timedelta(seconds=time_window)
        console.log(f"[red]Time window[/] (days, seconds, microseconds): {time_window}")
    console.log(f"Number of users retrieved: {len(users)}")
    return users


def compute_graph_from_users(users: List[UserModel]) -> List[dict]:
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
    return edges, edge_weights


@app.command()
def message(
    user_id: Annotated[str, typer.Argument(help="user-id to select")],
    user_message: Annotated[
        str, typer.Option(..., "-m", "--message", help="message to send to user.")
    ] = DEFAULT_MESSAGE,
):
    """Message a user by their user-id."""
    try:
        _ = UserModel.get(user_id)
    except Exception as e:
        console.log(f"An error occurred: {e}")
        return
    queue_message_to_user(user_id, user_message, "message_processing_queue")


@app.command("display-by-rank")
def display_by_rank(
    number_of_users: Annotated[int, typer.Option("-n", "--number", help="number of users to display")] = 7,
    delta_seconds: Annotated[int, typer.Option("-t", "--time", help="time window in seconds")] = -1,
) -> None:
    """Select the best connected node and save the conversation to a file by rank."""
    if delta_seconds > 0:
        console.log(f"Selecting users with last update within {delta_seconds} seconds...")
    users = get_users(group_id, platform_id, last_user_update_delta_seconds=delta_seconds)
    if len(users) > 1:
        edges, _ = compute_graph_from_users(users)
        best_nodes = select_best_connected_nodes(edges, num=number_of_users)
        # display if user was flagged already
        for i, node in enumerate(best_nodes):
            user = UserModel.get(node)
            console.print(f"{i+1}: {node} - {user.flagged}")
    elif len(users) == 1:
        user = users[0]
        console.print(f"1: {user.id} - {user.flagged}")
    else:
        console.print("No users found!")


@app.command()
def select(
    user_id: Annotated[str, typer.Argument(help="user-id to select")],
) -> None:
    """Select the user and do all the great things..."""

    # Check if user is flagged (accepted invite)
    try:
        user = UserModel.get(user_id)
    except Exception:
        console.log(f"User-id [red]{user_id}[/] not found! Exiting...")
        return
    if not user.flagged:
        console.log(f"User {user_id} is not flagged!")

    # Save the conversation to a file
    try:
        save_path = root_path / "data/conversations" / f"{group_id}-{platform_id}" / "output.txt"
        if not os.path.exists(save_path.parent):
            os.makedirs(save_path.parent)
        conversation = extract_conversation_from_user(user_id)
        with open(save_path, "w") as f:
            f.write(conversation)
            console.log(f"Saved conversation from user {user_id} to: {save_path}")

        # Generate audio from the conversation and save it
        try:
            save_dir = root_path / "data/audio" / f"{group_id}-{platform_id}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            asyncio.run(generate_patternized_audio(conversation, save_dir))
            console.log(f"Saved audio from conversation to: {save_dir}")
        except Exception as e:
            console.log(e)

    except Exception as e:
        console.log(e)


@app.command()
def network(
    number_of_users: Annotated[
        int, typer.Option("-n", "--number", help="number of users to select based on interaction time")
    ] = -1,
) -> None:
    """Constuct and save network from all users"""
    users = get_users(group_id, platform_id, upper_bound_users=number_of_users)
    edges, edge_weights = compute_graph_from_users(users)
    construct_and_save_network(edges, edge_weights)


@app.command()
def plot(delta_seconds: Annotated[int, typer.Option("-t", "--time", help="time window in seconds")] = -1) -> None:
    """Plot tSNE embeddings, 3D PCA embeddings and edge distribution to data/plots/{group_id}-{platform_id}"""
    console.log("Plotting tSNE embedding clustering...")
    users = get_users(group_id, platform_id, last_user_update_delta_seconds=delta_seconds)
    distances = calculate_distances_between_users(users)
    embeddings = [np.array(user.conversation_embedding).reshape(1, -1) for user in users]
    names = []
    for user in users:
        parts = user.id.split("-")
        name = parts[2]
        names.append(name)

    fig = plot_tSNE_embeddings(embeddings, 2, names, 15)
    save_dir = root_path / "data/plots" / f"{group_id}-{platform_id}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir / "tSNE_embeddings.png"
    fig.savefig(save_path)
    console.log(f"Saved tSNE plot to: {save_path}")

    fig = plot_3d_pca_embeddings(users, names)
    save_path = save_dir / "PCA_embeddings.png"
    fig.savefig(save_path)
    console.log(f"Saved 3d-PCA plot to: {save_path}")

    fig = plot_edge_distribution(distances)
    save_path = save_dir / "edge_distribution.png"
    fig.savefig(save_path)
    console.log(f"Saved edge distribution plot to: {save_path}")


@app.command(name="save-local")
def save_local(
    replace_flag: Annotated[bool, typer.Option("-r", "--replace", help="save local images to data/images")] = False,
    number_of_users: Annotated[int, typer.Option("-n", "--number", help="number of users to save")] = 15,
    crop_image_flag: Annotated[bool, typer.Option("-c", "--crop", help="crop image to circle")] = False,
    save_video_flag: Annotated[bool, typer.Option("-v", "--video", help="save video to data/videos")] = False,
) -> None:
    """Save images, network and videos locally."""
    users = get_users(group_id, platform_id, upper_bound_users=number_of_users)

    # Network
    console.log("Computing network graph to save...")
    edges, edge_weights = compute_graph_from_users(users)
    construct_and_save_network(edges, edge_weights)

    # Images
    save_dir = root_path / "data/images" / f"{group_id}-{platform_id}"
    # Delete the existing directory
    if replace_flag:
        if os.path.exists(save_dir):
            os.system(f"rm -r {save_dir}")
            console.log(f"[red]Deleted existing directory:[/] {save_dir}")

    relative_dir = Path("data/images") / f"{group_id}-{platform_id}"
    console.log(f"Saving images to local directory: {relative_dir} for {len(users)} users")
    for user in users:
        try:
            image = return_image_from_user(user, crop=crop_image_flag)
            # make a directory in data/images if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Save the image to a file
            save_path = save_dir / f"{user.id}.png"
            relative_save_path = relative_dir / f"{user.id}.png"
            image.save(save_path)
            console.log(f"[green]Saved image to[/]: {relative_save_path}")
        except Exception as e:
            console.log(e)

    # Videos
    if save_video_flag:
        console.log("Saving videos to local directory...")
        save_dir = root_path / "data/videos" / f"{group_id}-{platform_id}"
        relative_dir = Path("data/videos") / f"{group_id}-{platform_id}"
        for user in users:
            video_user = UserVideoModel.get(user.id)
            video_bytes = video_user.avatar_video_bytes
            video_path = save_dir / f"{video_user.id}.mp4"
            with open(video_path, "wb") as f:
                f.write(base64.b64decode(video_bytes))
            console.log(f"[green]Saved video to[/]: {relative_dir / f'{video_user.id}.mp4'}")

    console.log("Done!")


@app.command()
def video() -> None:
    # video_users = UserVideoModel.find(
    #     (UserVideoModel.group_id == group_id) & (UserVideoModel.platform_id == platform_id)
    # ).all()
    # for video_user in video_users:
    #     video_user.delete()
    users = get_users(group_id, platform_id)
    video_path = (
        "/Users/eric/Library/CloudStorage/Dropbox/git/github/svaeva/svaeva_eric/svaeva-evaluation/data/videos/video.mp4"
    )
    # load video
    with open(video_path, "rb") as f:
        video_bytes = base64.b64encode(f.read()).decode("utf-8")
    for user in users:
        print(user.id)
        video_user = UserVideoModel(
            id=user.id,
            avatar_video_bytes=video_bytes,
        )
        video_user.save()


@app.command()
def version() -> None:
    """Print the version of the package."""
    from svaeva_evaluation import version

    console.print(f"[yellow]svaeva-evaluation[/] version: [bold blue]{version}[/]")
    raise typer.Exit()


if __name__ == "__main__":
    app()
