import json

from src.our_chessboard_detection import (read_image, apply_filters, get_contours, 
                                      rotate_and_crop, align_board,chesboard_grids, 
                                      process_chessboard, display_chessboard_squares,
                                      offset_piece_coords, reverse_piece_coordinates)

def process_image(image_path):

    image = read_image(image_path, False)
    filtered_images = apply_filters(image, False)
    chess_contour = get_contours(filtered_images, show=False,  kernel_size=(25,25) ,  kernel_usage=True, iterations=4)
    warped_image, M = rotate_and_crop(filtered_images, chess_contour[0][1], show=False)
    rotated_image, best_angle = align_board(warped_image, radius=12, angle_step=90, show=False)
    grid_image = chesboard_grids(rotated_image, show = False)
    squares = display_chessboard_squares(rotated_image, show = False)
    board_matrix, piece_coords, pieces_count = process_chessboard(squares)
    piece_coords_global = offset_piece_coords(piece_coords, board_matrix,)

    # Reverse transforms
    original_coords = reverse_piece_coordinates(
        piece_coords_global,
        rotation_angle=best_angle,  # from align_board
        perspective_matrix=M,       # from rotate_and_crop
        rotated_image_shape=rotated_image.shape
    )
    print(original_coords)


    # Convert the board matrix to a compact nested list format
    board_matrix_list = board_matrix.tolist()

    # Convert original_coords to a JSON-serializable format
    original_coords_list = [dict(coord) for coord in original_coords]

    return {
        "image_path": image_path,
        "num_pieces": pieces_count,
        "board": board_matrix_list,
        "detected_pieces": original_coords_list
    }

def generate_output(input_file_path, output_file_path='output.json'):
    with open(input_file_path, 'r') as infile:
        data = json.load(infile)

    results = []
    for image_path in data.get('image_files', []):
        result = process_image(image_path)
        result['image'] = image_path
        results.append(result)

    with open(output_file_path, 'w') as outfile:
        json.dump(results, outfile, cls=CompactBoardJSONEncoder)



import json

class CompactBoardJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs['indent'] = 4  # Still keep overall formatting
        super().__init__(*args, **kwargs)

    def encode(self, o):
        if isinstance(o, list):
            # Compact encoding for lists of integers
            if all(isinstance(i, list) and all(isinstance(j, int) for j in i) for i in o):
                return "[\n" + ",\n".join(
                    ["    " + json.dumps(row) for row in o]
                ) + "\n]"
        return super().encode(o)


if __name__ == "__main__":
    # Example usage
    # process_image('image.jpg')
    generate_output('json_requests/input.json', 'output.json')

