from kg.kg_utils import (
    load_graph,
    get_neighbors,
    find_path,
)


if __name__ == "__main__":
    kg = load_graph()

    print("Neighbors of Intersection 4:")
    print(kg.get_neighbors("Intersection 4"))

    print("\nPath from Intersection 4 to Central Hospital:")
    print(kg.find_path("Intersection 4", "Central Hospital"))

    print("\nNearest hospital to Intersection 4:")
    print(kg.find_nearest_node_of_type("Intersection 4", "hospital"))
