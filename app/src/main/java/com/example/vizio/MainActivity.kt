package com.example.vizio

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.core.content.ContextCompat
import com.example.vizio.ml.Vizio8
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.Locale
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {

    private var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)

    private val CAMERA_REQUEST_CODE = 101
    lateinit var cameraManager: CameraManager
    lateinit var cameraDevice: CameraDevice
    lateinit var textureView: TextureView
    lateinit var imageView: ImageView
    lateinit var inputMic: ImageView
    lateinit var handler: Handler
    lateinit var bitmap: Bitmap
    lateinit var model: Vizio8
    lateinit var imageProcessor: ImageProcessor
    lateinit var labels: List<String>
    private val paint = Paint()

    private var tts: TextToSpeech? = null
    private var lastSpokenLabel: String? = null
    private var lastSpokenTime: Long = 0
    private var DEBOUNCE_TIME = 4000 // initial 4 seconds
    private var speak = true

    private val REQUEST_CODE_SPEECH_INPUT = 1
    private var currentUnit: String = "inches"

    companion object {
        const val ENTRANCE_WIDTH = 60.666666666666664 //inches
        const val ESCALATOR_WIDTH = 56.666666666666664 //inches
        const val STAIR_WIDTH = 72.33333333333333 //inches

        const val FOCAL_ENTRANCE_FRONT = 74.72527472527473
        const val FOCAL_ENTRANCE_SIDE = 48.57142857142858
        const val FOCAL_ESCALATOR_FRONT = 89.88235294117646
        const val FOCAL_ESCALATOR_SIDE = 79.76470588235294
        const val FOCAL_STAIR_FRONT = 76.68202764976958
        const val FOCAL_STAIR_SIDE = 70.23041474654379

    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels=FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize by scaling with 1/255
            .build()
        model = Vizio8.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        imageView = findViewById(R.id.ImageView)
        textureView = findViewById(R.id.textureView)
        inputMic = findViewById(R.id.inputMic)
        inputMic.setOnClickListener {
            val utteranceId = "prompt_utterance_id"
            tts?.speak("How can Vizio help you?", TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        }

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                tts = TextToSpeech(this@MainActivity) { status ->
                    if (status != TextToSpeech.ERROR) {
                        val result = tts?.setLanguage(Locale.US)
                        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                            // Notify the user that the language data is missing or not supported
                            Log.e("TTS", "Language not supported or missing data")

                        } else {
                            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                                override fun onStart(utteranceId: String?) {
                                    // Called when the TTS starts speaking
                                }

                                override fun onDone(utteranceId: String?) {
                                    // Called when the TTS finishes speaking
                                    startSpeechRecognition()
                                }

                                override fun onError(utteranceId: String?) {
                                }
                            })
                        }
                    } else {
                        Log.e("TTS", "Initialization failed")
                    }
                }
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
//                bitmap = Bitmap.createScaledBitmap(bitmap, 320, 320,true)

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 320, 320, 3), DataType.FLOAT32)
                var tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(bitmap)
                tensorImage = imageProcessor.process(tensorImage)
                inputFeature0.loadBuffer(tensorImage.buffer)

                val outputs = model.process(inputFeature0)
                val scores = outputs.outputFeature0AsTensorBuffer.floatArray
                val locations = outputs.outputFeature1AsTensorBuffer.floatArray
                val numberOfDetections = outputs.outputFeature2AsTensorBuffer.floatArray
                val classes = outputs.outputFeature3AsTensorBuffer.floatArray

                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width

                paint.textSize = h/25f
                paint.strokeWidth = h/95f
                var x = 0

                scores.forEachIndexed { index, score ->
                    x = index
                    x *= 4

                    if(score > 0.7){
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE

                        // Multiply 320x320 resolution
                        val xmin = locations[x + 1] * 320
                        val xmax = locations[x + 3] * 320

                        val objectWidthInFrame = (xmax - xmin).toDouble()

                        Log.d("xmin: ",  xmin.toString())
                        Log.d("xmax: ",  xmax.toString())
                        Log.d("w: ",  w.toString())
                        Log.d("Width in frame: ",  objectWidthInFrame.toString())

                        canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
                        paint.style = Paint.Style.FILL

                        val rawLabel = labels.get(classes.get(index).toInt())
                        val formattedLabel = formatLabel(rawLabel)


//                    Log.d("scores", "1 : "
//                            + scores[0].toString() + "," + scores[1].toString() +"," + scores[2].toString() +"," + scores[3].toString()
//                            + scores[4].toString() + "," + scores[5].toString() +"," + scores[6].toString() +"," + scores[7].toString()
//                            + scores[8].toString() + "," + scores[9].toString()
//                    )
//                    Log.d("locations", "2 : " + locations[0].toString())
//                    Log.d("numberOfDetections", "3 : " + numberOfDetections[0].toString())
//                    Log.d("classes", "4 : " + classes[0].toString())


                        var distance: Double = 0.0
                        var audioFeedback = ""
                        var textFeedback = ""

                        when (rawLabel) {
                            "entrance_front" -> {
                                distance = distanceFinder(FOCAL_ENTRANCE_FRONT, ENTRANCE_WIDTH, objectWidthInFrame)
                            }
                            "entrance_left", "entrance_right" -> {
                                distance = distanceFinder(FOCAL_ENTRANCE_SIDE, ENTRANCE_WIDTH, objectWidthInFrame)
                            }

                            "escalator_front" -> {
                                distance = distanceFinder(FOCAL_ESCALATOR_FRONT, ESCALATOR_WIDTH, objectWidthInFrame)
                            }
                            "escalator_left", "escalator_right" -> {
                                distance = distanceFinder(FOCAL_ESCALATOR_SIDE, ESCALATOR_WIDTH, objectWidthInFrame)
                            }

                            "stair_front" -> {
                                distance = distanceFinder(FOCAL_STAIR_FRONT, STAIR_WIDTH, objectWidthInFrame)
                            }
                            "stair_left", "stair_right" -> {
                                distance = distanceFinder(FOCAL_STAIR_SIDE, STAIR_WIDTH, objectWidthInFrame)
                            }
                        }

                        val roundedScore = String.format("%.2f", score * 100)

                        val normalizedUnit = when (currentUnit) {
                            "inch" -> "inches"
                            "foot" -> "feet"
                            "metre" -> "metres"
                            "centimetre" -> "centimetres"
                            "millimetre" -> "millimetres"
                            else -> currentUnit
                        }

                        audioFeedback = when (normalizedUnit) {
                            "inches" -> "$formattedLabel ${distance.roundToInt()} inches away"
                            "feet" -> "$formattedLabel ${(distance / 12.0).roundToInt()} feet away"
                            "metres" -> "$formattedLabel ${String.format("%.2f", distance * 0.0254)} metres away"
                            "centimetres" -> "$formattedLabel ${(distance * 2.54).roundToInt()} centimetres away"
                            "millimetres" -> "$formattedLabel ${(distance * 25.4).roundToInt()} millimetres away"
                            else -> "$formattedLabel ${distance.roundToInt()} inches away"
                        }

                        textFeedback = when (normalizedUnit) {
                            "inches" -> "$rawLabel $roundedScore% ${distance.roundToInt()} inches away"
                            "feet" -> "$rawLabel $roundedScore% ${(distance / 12.0).roundToInt()} feet away"
                            "metres" -> "$formattedLabel ${String.format("%.2f", distance * 0.0254)} metres away"
                            "centimetres" -> "$rawLabel $roundedScore% ${(distance * 2.54).roundToInt()} centimetres away"
                            "millimetres" ->"$rawLabel $roundedScore% ${(distance * 25.4).roundToInt()} millimetres away"
                            else -> "$rawLabel $roundedScore% ${distance.roundToInt()} inches away"
                        }

                        canvas.drawText(rawLabel, locations.get(x+1)*w, locations.get(x)*h, paint)

                        // Check if the text-to-speech engine is currently speaking.
                        val isTTSSpeaking = tts?.isSpeaking ?: false

                        // Check if the text-to-speech engine is currently speaking and if it should speak
                        if (!isTTSSpeaking && speak) {
                            val currentTime = System.currentTimeMillis()
                            if (rawLabel != lastSpokenLabel || (currentTime - lastSpokenTime) > DEBOUNCE_TIME) {
                                tts?.speak(audioFeedback, TextToSpeech.QUEUE_FLUSH, null, null)
                                lastSpokenLabel = rawLabel
                                lastSpokenTime = currentTime
                            }
                        }

                        Toast.makeText( this@MainActivity, textFeedback, Toast.LENGTH_SHORT).show()
                    }
                }

                imageView.setImageBitmap(mutable)

            }

        }

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

    }

    private fun formatLabel(rawLabel: String): String {
        val parts = rawLabel.split('_')
        if (parts.size == 2) {
            val objectPart = parts[0].capitalize()
            val directionPart = parts[1].capitalize()
            return "$objectPart at $directionPart"
        }
        return rawLabel // return original label if it doesn't match expected format
    }

    private fun distanceFinder(focalLength: Double, realObjectWidth: Double, widthInFrame: Double): Double {
        return (realObjectWidth * focalLength) / widthInFrame
    }

    private fun startSpeechRecognition() {
        speak = false
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())

        try {
            startActivityForResult(intent, REQUEST_CODE_SPEECH_INPUT)
        } catch (e: Exception) {
            Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    data class VoiceCommand(val pattern: String, val handler: (MatchResult) -> Unit)

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_CODE_SPEECH_INPUT && resultCode == RESULT_OK && data != null) {
            val res: ArrayList<String> = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS) as ArrayList<String>
            val recognizedText = res[0].toLowerCase(Locale.ROOT)

            val voiceCommands = listOf(
                VoiceCommand("(?:change )?(?:unit of )?measurements? (?:to|2) (.+)") { match ->
                    currentUnit = match.groups[1]?.value ?: ""
                },
                VoiceCommand("(?:set )?(?:a )?(bounce|debounce|report) (rate|time) (?:to|2) (\\d+)(?: seconds)?") { match ->
                    val newRate = match.groups[3]?.value?.toIntOrNull()
                    if (newRate != null && newRate > 0) {
                        DEBOUNCE_TIME = newRate * 1000
                    } else {
                        speak = true
                        tts?.speak("Invalid rate.", TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                }
            )

            var matched = false
            for (command in voiceCommands) {
                val regex = command.pattern.toRegex()
                val matchResult = regex.find(recognizedText)
                if (matchResult != null) {
                    command.handler(matchResult)
                    matched = true
                    speak = true
                    tts?.speak("Your command will be executed.", TextToSpeech.QUEUE_FLUSH, null, null)
                    break
                }
            }

            if (!matched) {
                speak = true
                tts?.speak("Unrecognized command. Please try again.", TextToSpeech.QUEUE_FLUSH, null, null)
            }

            speak = true
        }
    }

    @SuppressLint("MissingPermission")
    private fun open_camera(){
        cameraManager.openCamera(cameraManager.cameraIdList[0], object: CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0

                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)

                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)
            }

            override fun onDisconnected(p0: CameraDevice) {

            }

            override fun onError(p0: CameraDevice, p1: Int) {

            }
        }, handler)
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
        model.close()
    }

    private fun get_permission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), CAMERA_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (grantResults.isNotEmpty() && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            get_permission()
        }
    }

}