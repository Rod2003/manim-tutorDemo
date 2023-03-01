from manim import *
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
from manim_ml.neural_network.animations.dropout import make_neural_network_dropout_animation

config.pixel_height = 800
config.pixel_width = 1000
config.frame_height = 8.0
config.frame_width = 8.0

# Define scene
class NNScene(Scene):
    # Generate scene
    def construct(self):
        # Make nn1
        nn1 = NeuralNetwork([
                FeedForwardLayer(3),
                FeedForwardLayer(5),
                FeedForwardLayer(3),
            ],
            layer_spacing=0.6,
        )
        # Position nn1
        nn1.move_to(2 * LEFT)

        # Make nn2
        nn2 = NeuralNetwork([
                FeedForwardLayer(3),
                FeedForwardLayer(5),
                FeedForwardLayer(3),
            ],
            layer_spacing=0.6,
        )
        # Position nn2
        nn2.move_to(2 * RIGHT)

        # Circles represent output
        # Make circles
        c1 = Circle(radius=0.4, color=BLUE_B, fill_opacity=1)
        c2 = Circle(radius=0.4, color=YELLOW_A, fill_opacity=1)

        # Group circles
        VGroup(c1, c2).set_x(0).arrange(buff=2).set_y(-2)

        # Tutor.AI logo image
        tutor = SVGMobject("media/images/container-accent-circle-border-1.svg")
        tutor.scale(0.5)
        tutor.move_to(ORIGIN)

        # ANIMATIONS
        self.play(FadeIn(nn1))
        self.play(
            make_neural_network_dropout_animation(
                nn1, dropout_rate=0.25, do_forward_pass=True, first_layer_stable=True, seed=3
            )
        )

        self.play(GrowFromPoint(c1, nn1.get_right()))
        self.wait(0.5)

        self.play(FadeIn(nn2))
        self.play(
            make_neural_network_dropout_animation(
                nn2, dropout_rate=0.25, do_forward_pass=True, first_layer_stable=True, seed=2
            )
        )

        self.play(GrowFromPoint(c2, nn2.get_right()))
        self.wait(1)

        # Matching animation
        self.play(c1.animate.shift(1.4 * RIGHT + 2 * UP), c2.animate.shift(1.4 * LEFT + 2 * UP))
        self.play(FadeOut(c1), FadeOut(c2))

        self.clear()

        self.play(FadeIn(tutor))

        self.wait(2)

