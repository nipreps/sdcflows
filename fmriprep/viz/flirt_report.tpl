<div class="flirt_report">
    <h4> FLIRT </h4>
    <style type="text/css">
        @keyframes flickerAnimation {
        0% {
            opacity: 1;
        }
        100% {
            opacity: 0;
        }
        }
        .flirt_report_image div {
            animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation;
        }

    </style>
    <div class="flirt_report_image" style="background-image: url(data:image/svg+xml;utf8,{{ reference_image }}); background-repeat: no-repeat; backgroundSize: 100%;">
        <div>
            {{output_image }}
        </div>
    </div>
</div>
