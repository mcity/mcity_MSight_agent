"""Standalone MSight node (not an MCP tool, launched as a subprocess by msight_record_archive.py): draws detection boxes and republishes frames as ImageData for image_to_video_aggregator to consume."""
import cv2

from msight_core.data import DetectionResultsData, ImageData
from msight_core.nodes import DataProcessingNode
from msight_core.nodes.base import NodeConfig


class AnnotatedFramePublisherNode(DataProcessingNode):
    default_configs = NodeConfig(
        subscribe_topic_data_type=DetectionResultsData,
        publish_topic_data_type=ImageData,
    )

    def process(self, data: DetectionResultsData):
        frame = data.raw_sensor_data.to_ndarray().copy()
        for obj in data.detection_result.object_list:
            x1, y1, x2, y2 = map(int, obj.box)
            px, py = map(int, obj.pixel_bottom_center)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            label = f"{obj.class_id} {obj.score:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return ImageData.from_ndarray(frame, sensor_name=data.sensor_name)


def main():
    from msight_core.utils import get_default_arg_parser, get_node_config_from_args

    parser = get_default_arg_parser(
        description="Republish detection-annotated frames (bounding boxes drawn) for recording.",
        node_class=AnnotatedFramePublisherNode,
    )
    args = parser.parse_args()
    configs = get_node_config_from_args(args)
    AnnotatedFramePublisherNode(configs).spin()


if __name__ == "__main__":
    main()
