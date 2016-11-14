<div id='{{unique_string}}'>
    <h4>{{unique_string}}</h4>
    <style type='text/css'>
    @keyframes flickerAnimation {
        0% {
            opacity: 1;
        }
        100% {
            opacity: 0;
        }
    }

    #{{unique_string}} .image_container {
        position: relative;
    }

    #{{unique_string}} .image_container .overlay_image {
        position: absolute;
            top: 0;
            let: 0;
        background-size: 100%;
        animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation;
    }
    </style>
    <div class='image_container'>
    <div class='base_image'>{{base_image}}</div>
    <div class='overlay_image'>{{overlay_image}}</div>
    </div>
</div>
