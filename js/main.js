// const videoElement = document.querySelector(".VideoPlayer-video-juqPz");
const videoElement = document.querySelector("video");

// 创建AudioContext

// 创建WebSocket连接
const socket = new WebSocket('ws://localhost:8765');

// let sourceNode;
// let processorNode;

const audioContext = new (window.AudioContext || window.webkitAudioContext)();
// const source = audioContext.createMediaElementSource(videoElement);


const captureStream = videoElement.captureStream();
console.log("captureStream", captureStream);

let recorder;
window.__recorder = recorder;
let audioChunks = [];



function sendData(data) {
    console.log('recorder.mimeType', recorder.mimeType)
    // const audioBlob = new Blob([data], { type: recorder.mimeType });
    const audioBlob = new Blob(data, { type: 'audio/webm' });

    const audioUrl = URL.createObjectURL(audioBlob);
    console.log("Audio recording saved.");
    socket.send(audioBlob);
    audioChunks = []
    // 创建下载链接
    // const link = document.createElement("a");
    // link.href = audioUrl;
    // link.download = "audio_recording.webm";
    // link.click();
    // setTimeout(() => {
    //     audioChunks = [];
    // }, 1000)

}

// document.querySelector("#stop").addEventListener("click", () => {
//     recorder.stop();
//     createDownload();
// });



// 接收并显示字幕
socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.subtitle) {
        console.log('subtitle:', data.subtitle);
        // set subtitle
        $d.innerText = data.subtitle
        // subtitleDiv.textContent = data.subtitle;
        // subtitleDiv.style.opacity = 1;

        // // 5秒后淡出字幕
        // setTimeout(() => {
        //     subtitleDiv.style.opacity = 0;
        // }, 5000);
    } else {
        $d.innerText = ''
    }
};

function startRecording() {
    const audioTracks = captureStream.getAudioTracks();
    //播放后获取audioTracks
    console.log("audioTracks", audioTracks);
    const audioStream = new MediaStream(audioTracks);

    recorder = new MediaRecorder(audioStream);

    recorder.ondataavailable = (event) => {
        console.log("event", event);
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
        // todo 优化
        // sendData(event.data);
    };
    recorder.onstop = () => {
        if (!audioChunks?.length) {
            return;
        }
        sendData(audioChunks);
    };

    recorder.start();
    timer = setInterval(() => {
        recorder.stop()
        recorder.start()
    }, 1500);
}

function stopRecording() {
    clearInterval(timer);
    recorder.stop();
}

let timer;
// // 确保在用户交互后开始处理音频
videoElement.addEventListener('play', () => {
    startRecording()
});

videoElement.addEventListener('pause', () => {
    stopRecording()
})

let $d
function createSubTitle() {
    $d = document.createElement('div')
    $d.style.position = 'absolute'
    $d.style.bottom = '20px'
    $d.style.left = '50%'
    $d.style.transform = 'translateX(-50%)'
    $d.style.color = 'white'
    $d.style.fontSize = '24px'
    $d.style.opacity = 1
    $d.style.backgroundColor = 'rgba(0,0,0,0.5)'
    $d.style.padding = '10px'
    $d.style.borderRadius = '5px'
    $d.style.zIndex = 9999
    $d.style.transition = 'opacity 1s'
    // append 相邻 video
    videoElement.parentElement.appendChild($d)
}
createSubTitle()

