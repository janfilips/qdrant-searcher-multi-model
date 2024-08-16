import json
import logging
import os
from typing import List

# import httpx
import requests
from fastembed import TextEmbedding
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.sparse.bm25 import Bm25
from httpx import HTTPStatusError, RequestError
from qdrant_client import QdrantClient, models
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from backend import config

logger = logging.getLogger(__name__)

YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
RED = "\033[1;31m"
GREEN = "\033[1;32m"
CYAN = "\033[1;36m"
RESET = "\033[0m"

client = QdrantClient(f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}", timeout=600)

# XXX these should be loaded from dependencies.py
dense_embedding_model = TextEmbedding("snowflake/snowflake-arctic-embed-l", cache_dir="tokenizer/snowflake-arctic-embed-l")
bm25_embedding_model = Bm25("Qdrant/bm25", cache_dir="tokenizer/qdrant_bm25")
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", cache_dir="tokenizer/colbert-ir_colbertv2.0")

CANDIDATES_COLLECTION_NAME = "neural_candidates"
# CANDIDATES_COLLECTION_NAME = "neural_candidates_with_skills_analysis"

# Note: A very good SPLADE transformer is also this one naver/efficient-splade-VI-BT-large-query
SPARSE_EMBEDING_MODEL = "bm25"
DENSE_EMBEDING_MODEL = "openai-embedding-3-small"
LATE_INTERACTION_EMBEDING_MODEL = "colbertv2.0"

logo = """
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗         ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗███████╗██████╗     ██╗   ██╗  ██████╗  ██████╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║         ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║██╔════╝██╔══██╗    ██║   ██║  ╚════██╗██╔═████╗
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║         ███████╗█████╗  ███████║██████╔╝██║     ███████║█████╗  ██████╔╝    ██║   ██║   █████╔╝██║██╔██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║         ╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║██╔══╝  ██╔══██╗    ╚██╗ ██╔╝  ██╔═══╝ ████╔╝██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗    ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║███████╗██║  ██║     ╚████╔╝██╗███████╗╚██████╔╝
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝      ╚═══╝ ╚═╝╚══════╝ ╚═════╝
"""  # noqa
logo_starter = """
.          .       _   _                        |"|           !!!         |                                         .      .    #   ___          _   _
.    .:::.         '\\-//`        ()_()         _|_|_       `  _ _  '      |.===.         ,,,,,         ()_()      .  .:::.      #  <_*_>         '\\-//`
    :(o o):  .     (o o)         (o o)         (o o)      -  (OXO)  -     {}o o{}       /(o o)\        (o o)        :(o o):  .  #  (o o)         (o o)
ooO--(_)--Ooo-ooO--(_)--Ooo-ooO--`o'--Ooo-ooO--(_)--Ooo-ooO--(_)--Ooo-ooO--(_)--Ooo-ooO--(_)--Ooo-ooO--`o'--Ooo-ooO--(_)--Ooo--8---(_)--Ooo-ooO--(_)--Ooo-
"""  # noqa

logo_dense_emebeddings_search = """
 ____  _____ _   _ ____  _____      _____ __  __ ____  _____ ____  ____ ___ _   _  ____ ____
|  _ \| ____| \ | / ___|| ____|    | ____|  \/  | __ )| ____|  _ \|  _ \_ _| \ | |/ ___/ ___|
| | | |  _| |  \| \___ \|  _|      |  _| | |\/| |  _ \|  _| | | | | | | | ||  \| | |  _\___ \\
| |_| | |___| |\  |___) | |___     | |___| |  | | |_) | |___| |_| | |_| | || |\  | |_| |___) |
|____/|_____|_| \_|____/|_____|    |_____|_|  |_|____/|_____|____/|____/___|_| \_|\____|____/
"""  # noqa

logo_sparse_embeddings_search = """
 ____  ____   _    ____  ____  _____      _____ __  __ ____  _____ ____  ____ ___ _   _  ____ ____
/ ___||  _ \ / \  |  _ \/ ___|| ____|    | ____|  \/  | __ )| ____|  _ \|  _ \_ _| \ | |/ ___/ ___|
\___ \| |_) / _ \ | |_) \___ \|  _|      |  _| | |\/| |  _ \|  _| | | | | | | | ||  \| | |  _\___ \\
 ___) |  __/ ___ \|  _ < ___) | |___     | |___| |  | | |_) | |___| |_| | |_| | || |\  | |_| |___)|
|____/|_| /_/   \_\_| \_\____/|_____|    |_____|_|  |_|____/|_____|____/|____/___|_| \_|\____|____/
"""  # noqa

logo_late_interaxtion_model_search = """
 _        _  _____ _____      ___ _   _ _____ _____ ____      _    ____ _____ ___ ___  _   _      __  __  ___  ____  _____ _
| |      / \|_   _| ____|    |_ _| \ | |_   _| ____|  _ \    / \  / ___|_   _|_ _/ _ \| \ | |    |  \/  |/ _ \|  _ \| ____| |
| |     / _ \ | | |  _|       | ||  \| | | | |  _| | |_) |  / _ \| |     | |  | | | | |  \| |    | |\/| | | | | | | |  _| | |
| |___ / ___ \| | | |___      | || |\  | | | | |___|  _ <  / ___ \ |___  | |  | | |_| | |\  |    | |  | | |_| | |_| | |___| |___
|_____/_/   \_\_| |_____|    |___|_| \_| |_| |_____|_| \_\/_/   \_\____| |_| |___\___/|_| \_|    |_|  |_|\___/|____/|_____|_____|
"""  # noqa

logo_rrf_starter = """
 ____  _____ ____ ___ ____  ____   ___   ____    _    _         ____      _    _   _ _  __     _____ _   _ ____ ___ ___  _   _
|  _ \| ____/ ___|_ _|  _ \|  _ \ / _ \ / ___|  / \  | |       |  _ \    / \  | \ | | |/ /    |  ___| | | / ___|_ _/ _ \| \ | |
| |_) |  _|| |    | || |_) | |_) | | | | |     / _ \ | |       | |_) |  / _ \ |  \| | ' /     | |_  | | | \___ \| | | | |  \| |
|  _ <| |__| |___ | ||  __/|  _ <| |_| | |___ / ___ \| |___    |  _ <  / ___ \| |\  | . \     |  _| | |_| |___) | | |_| | |\  |
|_| \_\_____\____|___|_|   |_| \_ \\___/ \____/_/   \_\_____|   |_| \_\/_/   \_\_| \_|_|\_\    |_|    \___/|____/___\___/|_| \_|
"""  # noqa

rrf_all_methods = """
 ____  ____  ____                     _    _     _          __  __ _____ _____ _   _  ___  ____  ____
|  _ \|  _ \|  _ \                   / \  | |   | |        |  \/  | ____|_   _| | | |/ _ \|  _ \/ ___|
| |_) | |_) | |_) |     _____       / _ \ | |   | |        | |\/| |  _|   | | | |_| | | | | | | \___ \\
|  _ <|  _ <|  __/     |_____|     / ___ \| |___| |___     | |  | | |___  | | |  _  | |_| | |_| |___) |
|_| \_\_| \_\_|                   /_/   \_\_____|_____|    |_|  |_|_____| |_| |_| |_|\___/|____/|____/
"""  # noqa

reranking_late_interaction_model = """
 ____  _____ ____      _    _   _ _  _____ _   _  ____      _        _  _____ _____      ___ _   _ _____ _____ ____      _    ____ _____ ___ ___  _   _
|  _ \| ____|  _ \    / \  | \ | | |/ /_ _| \ | |/ ___|    | |      / \|_   _| ____|    |_ _| \ | |_   _| ____|  _ \    / \  / ___|_   _|_ _/ _ \| \ | |
| |_) |  _| | |_) |  / _ \ |  \| | ' / | ||  \| | |  _     | |     / _ \ | | |  _|       | ||  \| | | | |  _| | |_) |  / _ \| |     | |  | | | | |  \| |
|  _ <| |___|  _ <  / ___ \| |\  | . \ | || |\  | |_| |    | |___ / ___ \| | | |___      | || |\  | | | | |___|  _ <  / ___ \ |___  | |  | | |_| | |\  |
|_| \_\_____|_| \_\/_/   \_\_| \_|_|\_\___|_| \_|\____|    |_____/_/   \_\_| |_____|    |___|_| \_| |_| |_____|_| \_\/_/   \_\____| |_| |___\___/|_| \_|
"""  # noqa

multistep_retrieval = """
 __  __ _   _ _   _____ ___ ____ _____ _____ ____       ____  _____ _____ ____  ___ _______     ___    _
|  \/  | | | | | |_   _|_ _/ ___|_   _| ____|  _ \     |  _ \| ____|_   _|  _ \|_ _| ____\ \   / / \  | |
| |\/| | | | | |   | |  | |\___ \ | | |  _| | |_) |    | |_) |  _|   | | | |_) || ||  _|  \ \ / / _ \ | |
| |  | | |_| | |___| |  | | ___) || | | |___|  __/     |  _ <| |___  | | |  _ < | || |___  \ V / ___ \| |___
|_|  |_|\___/|_____|_| |___|____/ |_| |_____|_|        |_| \_\_____| |_| |_| \_\___|_____|  \_/_/   \_\_____|
"""  # noqa


def create_collection(collection_name: str, dense_embeddings_size: int, late_interaction_embeddings_size: int) -> dict:
    qdrant_resp = client.create_collection(
        collection_name,
        vectors_config={
            DENSE_EMBEDING_MODEL: models.VectorParams(
                size=dense_embeddings_size,
                distance=models.Distance.COSINE,
            ),
            LATE_INTERACTION_EMBEDING_MODEL: models.VectorParams(
                size=late_interaction_embeddings_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            SPARSE_EMBEDING_MODEL: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )
    return qdrant_resp


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=3, min=3, max=5),
    retry=retry_if_exception_type((HTTPStatusError, RequestError))
)
def text_to_openai_vector(text: str, transformer_model_name: str = "text-embedding-3-small") -> List[float]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.OPENAI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": transformer_model_name
    }

    logger.info("Querying OpenAI for embeddings...")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()

        embedding = response.json()["data"][0]["embedding"]
        return embedding

    except HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        raise
    except RequestError as e:
        logger.error(f"Request error occurred: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")
        raise
    except KeyError as e:
        logger.error(f"KeyError in the response data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def get_text_embeddings(text: str) -> dict:
    # Dense embeddings
    dense_embeddings = text_to_openai_vector(text=text)

    # On-premise embeddings
    # dense_embeddings = list(dense_embedding_model.passage_embed(text))[0].tolist()

    # Sparse embeddings
    bm25_embeddings = list(bm25_embedding_model.passage_embed(text))[0].as_object()

    # Ensure BM25 is in correct format and convert numpy arrays to lists
    if isinstance(bm25_embeddings, dict):
        bm25_embeddings['indices'] = bm25_embeddings['indices'].tolist()
        bm25_embeddings['values'] = bm25_embeddings['values'].tolist()
    else:
        raise ValueError("BM25 embeddings are not correctly structured. Expected a dict with 'indices' and 'values'.")

    # Late interaction embeddings
    late_interaction_embeddings = list(late_interaction_embedding_model.passage_embed(text))[0].tolist()

    return {
        "dense_embeddings": dense_embeddings,
        "bm25_embeddings": bm25_embeddings,
        "late_interaction_embeddings": late_interaction_embeddings,

    }


def upload_vector(collection_name, point_id, payload, dense_embeddings, bm25_embeddings, late_interaction_embeddings) -> dict:
    qdrant_resp = client.upload_points(
        collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    DENSE_EMBEDING_MODEL: dense_embeddings,  # Should be a list
                    SPARSE_EMBEDING_MODEL: bm25_embeddings,  # Should be a dict with 'indices' and 'values' as lists
                    LATE_INTERACTION_EMBEDING_MODEL: late_interaction_embeddings,  # Should be a list
                },
                payload=payload
            )
        ],

        wait=True
    )
    return qdrant_resp


def map_candidate_data(candidate):
    candidate_language = candidate["user"]["prefered_language"]

    mapped_data = {
        "language": candidate_language.title(),
        "user_id": candidate["user_id"],
        "candidate_id": candidate["id"],
        "external_candidate_id": candidate["external_candidate_id"],
        "name": candidate["user"]["name"],
        "email": candidate["user"]["email"],
        "headline": candidate["bio"].get(candidate_language.lower(), {}).get("headline", ""),
        "expertise": candidate["bio"].get("english", {}).get("expertise", ""),
        "skill_tags": candidate["bio"].get("english", {}).get("skill_tags", []),
        "cv_summary": candidate["bio"].get("english", {}).get("cv_summary", ""),
        "skill_analysis": candidate["bio"].get("english", {}).get("skill_analysis", ""),
        "bio": candidate["bio"],
        "countries": ["XXX", "TODO"],
        "country": candidate["user"]["country"],
        "city": candidate["user"]["city"],
        "email_notifications_enabled": False,
        "notification_frequency": "never",
        "cv_url": candidate["uploaded_resumes"][0]["cv_url"],
    }

    return mapped_data


def recover_snapshot(location: str = "neural_candidates-multiple-representations.snapshot"):
    client.recover_snapshot(
        CANDIDATES_COLLECTION_NAME,
        location=location
    )


def search_candidates(query_text: str):

    text_embeddings = get_text_embeddings(text=query_text)

    dense_query_vector = text_embeddings["dense_embeddings"]
    sparse_query_vector = text_embeddings["bm25_embeddings"]
    late_query_vector = text_embeddings["late_interaction_embeddings"]

    # Extract the sparse vector and ensure it has the correct structure
    # sparse_vector_data = sparse_query_vector.as_object()

    # Ensure indices and values are lists
    sparse_vector_data = sparse_query_vector
    if not isinstance(sparse_vector_data["indices"], list):
        sparse_vector_data["indices"] = list(sparse_vector_data["indices"])
    if not isinstance(sparse_vector_data["values"], list):
        sparse_vector_data["values"] = list(sparse_vector_data["values"])

    ##################################################################################################################
    # Dense embeddings
    ##################################################################################################################
    print(logo_starter, logo_dense_emebeddings_search)
    print("*  DENSE EMBEDDINGS SEARCH **", "\n")

    limit = 10
    offset = 0
    total_results = 0

    MIN_SCORE_THRESHOLD = 0.6

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        # Perform the query
        qdrant_candidates = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            query=dense_query_vector,
            using=DENSE_EMBEDING_MODEL,
            with_payload=True,
            with_vectors=False,
            query_filter={},  # XXX TODO filter
            limit=limit,
            offset=offset,
            score_threshold=MIN_SCORE_THRESHOLD,
        )

        # Check if there are no more results
        if not qdrant_candidates.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in qdrant_candidates.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    ##################################################################################################################
    # Sparse embeddings
    ##################################################################################################################
    print(logo_starter, logo_sparse_embeddings_search)
    print("*  SPARSE EMBEDDINGS SEARCH  **", "\n")

    limit = 10
    offset = 0
    total_results = 0

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        results = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            query=models.SparseVector(**sparse_vector_data),
            using=SPARSE_EMBEDING_MODEL,
            query_filter={},  # XXX TODO filter
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        # Check if there are no more results
        if not results.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in results.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    ##################################################################################################################
    # Late interaction model
    ##################################################################################################################
    print(logo_starter, logo_late_interaxtion_model_search)
    print("*  LATE INTERACTION MODEL SEARCH ** ", "\n")

    limit = 10
    offset = 0
    total_results = 0

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        results = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            query=late_query_vector,
            using=LATE_INTERACTION_EMBEDING_MODEL,
            query_filter={},  # XXX TODO filter
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        # Check if there are no more results
        if not results.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in results.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    ##################################################################################################################
    # Reciprocal Rank Fusion (RRF) - Dense & Sparse embeddings
    ##################################################################################################################
    print(logo_starter, logo_rrf_starter)
    print("Reciprocal Rank Fusion (RRF) - Dense & Sparse embeddings", "\n")
    print("*  DENSE & SPARSE EMBEDDINGS ** ", "\n")

    limit = 10
    offset = 0
    total_results = 0

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=DENSE_EMBEDING_MODEL,
                limit=1_000,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vector_data["indices"], values=sparse_vector_data["values"]),
                using=SPARSE_EMBEDING_MODEL,
                limit=1_000,
            ),
        ]
        results = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            query_filter={},  # XXX TODO filter
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        # Check if there are no more results
        if not results.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in results.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    ##################################################################################################################
    # Reciprocal Rank Fusion (RRF) - All the methods in parallel
    ##################################################################################################################
    print(logo_starter, rrf_all_methods)
    print("Reciprocal Rank Fusion (RRF) - Dense & Sparse & Late Interaction embeddings", "\n")
    print("*  ALL METHODS IN PARALLEL ** ", "\n")

    limit = 10
    offset = 0
    total_results = 0

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=DENSE_EMBEDING_MODEL,
                limit=1_000
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vector_data["indices"], values=sparse_vector_data["values"]),
                using=SPARSE_EMBEDING_MODEL,
                limit=1_000
            ),
            models.Prefetch(
                query=late_query_vector,
                using=LATE_INTERACTION_EMBEDING_MODEL,
                limit=1_000
            )
        ]
        results = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            query_filter={},  # XXX TODO filter
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        # Check if there are no more results
        if not results.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in results.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    ##################################################################################################################
    # Reranking with late interaction model
    ##################################################################################################################
    print(logo_starter, reranking_late_interaction_model)
    print("* Reranking with late interaction model *\n")

    limit = 10
    offset = 0
    total_results = 0

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using=DENSE_EMBEDING_MODEL,
                limit=1_000
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vector_data["indices"], values=sparse_vector_data["values"]),
                using=SPARSE_EMBEDING_MODEL,
                limit=1_000
            )
        ]
        results = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            prefetch=prefetch,
            query=late_query_vector,
            using=LATE_INTERACTION_EMBEDING_MODEL,
            query_filter={},  # XXX TODO filter
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        # Check if there are no more results
        if not results.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in results.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    ##################################################################################################################
    # Multistep retrieval process
    ##################################################################################################################
    print(logo_starter, multistep_retrieval)
    print("* Multistep retrieval process *\n")

    limit = 10
    offset = 0
    total_results = 0

    # XXX temp
    test_candidates_count = 0

    count = 0
    force_stop = False
    while True:

        if force_stop:
            break

        results = client.query_points(
            CANDIDATES_COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=dense_query_vector,
                            # query=dense_query_vector.tolist(),
                            using=DENSE_EMBEDING_MODEL,
                            limit=1_000
                        )
                    ],
                    query=models.SparseVector(indices=sparse_vector_data["indices"], values=sparse_vector_data["values"]),
                    using=SPARSE_EMBEDING_MODEL,
                    limit=1_000
                ),
            ],
            query=late_query_vector,
            # query=late_query_vector.tolist(),
            using=LATE_INTERACTION_EMBEDING_MODEL,
            query_filter={},  # XXX TODO filter
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=offset,
        )

        # Check if there are no more results
        if not results.points:
            input("No more candidates found.")
            break

        # Process and print the results
        for candidate in results.points:
            score = candidate.score
            candidate = candidate.payload

            print(score, candidate["candidate_id"], candidate["headline"])
            total_results += 1

            # XXX temp
            if "test" in candidate["headline"].lower() or "qa" in candidate["headline"].lower():
                test_candidates_count += 1

            if count == 50:
                # XXX temp
                print("\ntest_candidates_count", test_candidates_count, "/", offset)
                user_input = input("\nMore results?? (yes/no): ").strip().lower()
                if user_input == "no" or user_input == "n":
                    force_stop = True
                    os.system("clear")
                    break
                count = 0
                print()
            count += 1

        # Update offset to retrieve the next batch of results
        offset += limit

    thoughts = input("\nThoughts? ")
    if thoughts == "shit":
        print("\nfuck\n")
    if thoughts == "fuck":
        print("\nshit\n")
    print()


def prep_candidate_text(candidate: dict) -> str:
    bio = candidate["bio"]["english"]
    job_titles = [job["job_title"] for job in bio["work_experience"]]

    candidate_text = bio["headline"] + ", " + ", ".join(job_titles) + "\n"
    candidate_text += bio["cv_summary"] + "\n"
    candidate_text += ", ".join(bio["expertise"]) + "\n"
    candidate_text += ", ".join(bio["skill_tags"]) + ", " + bio["comprehensive_skill_tags"]
    candidate_text += bio["skill_analysis"]

    return candidate_text


def insert_candidates_to_qdrant():

    collection_info = client.get_collection(CANDIDATES_COLLECTION_NAME)
    print("Collection info", collection_info)

    skip = 0
    limit = 100

    candidates_url = f"{config.PLATFORM_API_BASE_URL}"
    candidates_url += "/api/v1/candidates/all?skip={}&limit={}&ascending=true"

    count = 1
    while True:
        url = candidates_url.format(skip, limit)
        print(f"Fetching candidates {url}")
        response = requests.get(url, headers={'accept': 'application/json'})

        if response.status_code == 200:
            candidates = response.json()

            if candidates:
                for candidate in candidates:

                    mapped_candidate = map_candidate_data(candidate)

                    candidate_text = prep_candidate_text(mapped_candidate)

                    candidate_id = candidate["id"]
                    candidate_headline = candidate["bio"]["english"]["headline"]
                    candidate_embeddings = get_text_embeddings(text=candidate_text)

                    candidate_name = candidate["user"]["name"]
                    print(f"{count} Adding candidate {candidate_id} {candidate_name} {candidate_headline}")

                    upload_vector(
                        collection_name=CANDIDATES_COLLECTION_NAME,
                        payload=mapped_candidate,
                        point_id=candidate["id"],
                        dense_embeddings=candidate_embeddings["dense_embeddings"],
                        bm25_embeddings=candidate_embeddings["bm25_embeddings"],
                        late_interaction_embeddings=candidate_embeddings["late_interaction_embeddings"]
                    )
                    count += 1

            else:
                logger.info(f"{YELLOW}No candidates found. Retrying with skip={skip + limit}{RESET}")
        else:
            logger.error(f"{RED}Failed to fetch data. Status code: {response.status_code}{RESET}")
            break  # Exit the loop on error

        skip += limit


if __name__ == "__main__":

    # Create collection and insert candidates..
    # create_collection(collection_name=CANDIDATES_COLLECTION_NAME, dense_embeddings_size=1536, late_interaction_embeddings_size=128)
    # insert_candidates_to_qdrant()

    # ..or recover candidates from snapshot
    # recover_snapshot(location="neural_candidates-multiple-representations.snapshot")

    # os.system("clear")
    # print(logo)

    # query_text = "Tosca Testsuite, Api Testing, Automated Testing, Automation Testing Tools, Regression Testing, Test Automation Strategy, Requirements Gathering, Test Script Design And Implementation, Integration With Tools, Cross-Functional Collaboration"  # noqa
    # query_text = "Test Automation Engineer, TOSCA Testsuite, API Testing, Automated Testing, Automation Testing Tools, Regression Testing, Test Automation Strategy, English Language Proficiency, German Language Proficiency"
    query_text = "Test Automation Engineer Quality Engineering Test Automation TOSCA Testsuite API Testing Automated Testing Automation Testing Tools Regression Testing Test Automation Strategy English Language Proficiency German Language Proficiency"
    search_candidates(query_text=query_text)
