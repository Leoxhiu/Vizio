package com.example.vizio

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.TextureView
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.core.content.ContextCompat
import com.example.vizio.MainActivity.Companion.INCH_TO_CM_CONVERSION
import com.example.vizio.MainActivity.Companion.INCH_TO_METRE_CONVERSION
import com.example.vizio.MainActivity.Companion.INCH_TO_MM_CONVERSION
import com.example.vizio.ml.Vizio12
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

    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)

    private lateinit var layoutInitializer: LayoutInitializer

    val CAMERA_REQUEST_CODE = 101
    lateinit var cameraManager: CameraManager
    lateinit var cameraHandler: CameraHandler
    lateinit var handler: Handler
    lateinit var bitmap: Bitmap
    lateinit var model: Vizio12
    lateinit var imageProcessor: ImageProcessor
    lateinit var labels: List<String>
    val paint = Paint()

    var tts: TextToSpeech? = null
    var lastSpokenLabel: String? = null
    var lastSpokenTime: Long = 0
    var speak = true
    private val REQUEST_CODE_SPEECH_INPUT = 1

    var scoreThreshold = 90 // initial 90%
    var distanceThreshold = 2.2 // initial 2.2
    var debounceTime = 1000 // initial 1 seconds
    var currentUnit: String = "inches" // initial measurement
    var warningEnabled = false // initial warning


    companion object {
        const val INCH_TO_METRE_CONVERSION = 0.0254
        const val INCH_TO_CM_CONVERSION = 2.54
        const val INCH_TO_MM_CONVERSION = 25.4

        const val ENTRANCE_WIDTH = 60.666666666666664 //inches
        const val ESCALATOR_WIDTH = 56.666666666666664 //inches
        const val STAIR_WIDTH = 72.33333333333333 //inches

        const val FOCAL_ENTRANCE_FRONT = 72.96703296703298
        const val FOCAL_ENTRANCE_SIDE = 53.62637362637362
        const val FOCAL_ESCALATOR_FRONT = 83.29411764705883
        const val FOCAL_ESCALATOR_SIDE = 73.41176470588236
        const val FOCAL_STAIR_FRONT = 74.10138248847927
        const val FOCAL_STAIR_SIDE = 66.17511520737327
    }

    val labelMapping = mapOf(
        "entrance_front" to "entrance in front",
        "entrance_left" to "entrance on left",
        "entrance_right" to "entrance on right",
        "escalator_front" to "escalator in front",
        "escalator_left" to "escalator on left",
        "escalator_right" to "escalator on right",
        "stair_front" to "stair in front",
        "stair_left" to "stair on left",
        "stair_right" to "stair on right"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()

        labels=FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(320, 320, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize by scaling with 1/255
            .build()
        model = Vizio12.newInstance(this)
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)
        cameraHandler = CameraHandler(this, handler)
        loadPreferences()

        layoutInitializer = LayoutInitializer(this)
        layoutInitializer.initialize()
    }

    fun savePreferences() {
        val sharedPref = getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)
        with (sharedPref.edit()) {
            putInt("scoreThreshold", scoreThreshold)
            putFloat("distanceThreshold", distanceThreshold.toFloat())
            putInt("debounceTime", debounceTime)
            putString("currentUnit", currentUnit)
            putBoolean("warningEnabled", warningEnabled)
            apply()
        }
    }

    private fun loadPreferences() {
        val sharedPref = getSharedPreferences("AppPreferences", Context.MODE_PRIVATE)
        scoreThreshold = sharedPref.getInt("scoreThreshold", 90)
        distanceThreshold = sharedPref.getFloat("distanceThreshold", 2.2f).toDouble()
        debounceTime = sharedPref.getInt("debounceTime", 1000)
        currentUnit = sharedPref.getString("currentUnit", "inches") ?: "inches"
        warningEnabled = sharedPref.getBoolean("warningEnabled", false)
    }


    fun distanceFinder(focalLength: Double, realObjectWidth: Double, widthInFrame: Double): Double {
        return ((realObjectWidth * focalLength) / widthInFrame) * distanceThreshold
    }

    fun initializeTTS() {
        tts = TextToSpeech(this) { status ->
            if (status != TextToSpeech.ERROR) {
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    // Notify the user that the language data is missing or not supported
                    Log.e("TTS", "Language not supported or missing data")

                } else {
                    tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            // Called when the TTS starts speaking
                            initializeSpeechRecognition()
                        }

                        override fun onDone(utteranceId: String?) {
                        }

                        override fun onError(utteranceId: String?) {
                        }
                    })
                }
            } else {
                Log.e("TTS", "Initialization failed")
            }
        }
    }

    fun initializeSpeechRecognition() {
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
            val recognizedText = res[0].lowercase(Locale.ROOT)

            val voiceCommands = listOf(
                VoiceCommand("(?:change )?(?:unit of )?measurements? (?:to|2) (inch|inches|foot|feet|meter|meters|metre|metres|centimeter|centimeters|centimetre|centimetres|millimeter|millimeters|millimetre|millimetres)") { match ->
                    currentUnit = normalizeUnit(match.groups[1]?.value ?: "")
                    warningEnabled = false
                    savePreferences()
                },
                VoiceCommand("(no measurement|no measurements|disable measurement|disable measurements)") { _ ->
                    warningEnabled = true
                    savePreferences()
                },
                VoiceCommand("(?:set )?(?:a )?(bounce|debounce|report) (rate|time) (?:to|2) (\\d+)(?: seconds)?") { match ->
                    val newRate = match.groups[3]?.value?.toIntOrNull()
                    if (newRate != null && newRate > 0) {
                        debounceTime = newRate * 1000
                        savePreferences()
                    } else {
                        speak = true
                        tts?.speak("Invalid rate.", TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                },
                VoiceCommand("(thresholds? (?:change )?to (\\d{2}))") { match ->
                    val newValue = match.groups[2]?.value?.toIntOrNull()
                    if (newValue != null && newValue in 10..99) {
                        scoreThreshold = newValue
                        savePreferences()
                    } else {
                        speak = true
                        tts?.speak("Invalid threshold value.", TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                },
                VoiceCommand("(?:distance to ([0-4](?:\\.\\d{1,2})?|5))") { match ->
                    val newValue = match.groups[1]?.value?.toDoubleOrNull()
                    if (newValue != null) {
                        distanceThreshold = newValue
                        savePreferences()
                    } else {
                        speak = true
                        tts?.speak("Invalid distance value.", TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                }
            )

            var matched = false
            for (command in voiceCommands) {
                val regex = command.pattern.toRegex()
                val matchResult = regex.find(recognizedText)
                val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                if (matchResult != null) {
                    command.handler(matchResult)
                    matched = true
                    speak = true
                    tts?.speak("Your command will be executed.", TextToSpeech.QUEUE_FLUSH, null, null)
                    // Trigger haptic feedback
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
                    } else {
                        // For devices below API level 26
                        vibrator.vibrate(500)
                    }
                    break
                }
            }

            if (!matched) {
                speak = true
                tts?.speak("Unrecognised command. Please try again.", TextToSpeech.QUEUE_FLUSH, null, null)
            }

            speak = true

        } else if (resultCode == RESULT_CANCELED) {
            speak = true
        }
    }

    fun normalizeUnit(unit: String): String {
        return when (unit) {
            "inch", "inches" -> "inches"
            "foot", "feet" -> "feet"
            "meter", "meters", "metre", "metres" -> "metres"
            "centimeter", "centimeters", "centimetre", "centimetres" -> "centimetres"
            "millimeter", "millimeters", "millimetre", "millimetres" -> "millimetres"
            else -> unit
        }
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

class LayoutInitializer(private val activity: MainActivity) {

    val imageView: ImageView by lazy { activity.findViewById(R.id.ImageView) }
    val textureView: TextureView by lazy { activity.findViewById(R.id.textureView) }
    val inputArea: View by lazy { activity.findViewById(R.id.inputArea) }

    fun initialize() {

        val vibrator = activity.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        inputArea.setOnClickListener {
            // Trigger haptic feedback
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                // For devices below API level 26
                vibrator.vibrate(500)
            }

            val utteranceId = "prompt_utterance_id"
            activity.tts?.speak("How can Vizio help you?", TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        }

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {

                activity.initializeTTS()
                activity.cameraHandler.openCamera(activity.cameraManager, textureView)
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                activity.bitmap = textureView.bitmap!!
//                bitmap = Bitmap.createScaledBitmap(bitmap, 320, 320,true)

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 320, 320, 3), DataType.FLOAT32)
                var tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(activity.bitmap)
                tensorImage = activity.imageProcessor.process(tensorImage)
                inputFeature0.loadBuffer(tensorImage.buffer)

                val outputs = activity.model.process(inputFeature0)
                val scores = outputs.outputFeature0AsTensorBuffer.floatArray
                val locations = outputs.outputFeature1AsTensorBuffer.floatArray
                val numberOfDetections = outputs.outputFeature2AsTensorBuffer.floatArray
                val classes = outputs.outputFeature3AsTensorBuffer.floatArray

                var mutable = activity.bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width

                activity.paint.textSize = h/25f
                activity.paint.strokeWidth = h/95f
                var x = 0

                scores.forEachIndexed { index, score ->
                    x = index
                    x *= 4

                    if(score > (activity.scoreThreshold / 100.0)){
                        activity.paint.setColor(activity.colors.get(index))
                        activity.paint.style = Paint.Style.STROKE

                        // Find width in frame
                        // Multiply 320x320 resolution
                        val xmin = locations[x + 1] * 320
                        val xmax = locations[x + 3] * 320
                        val objectWidthInFrame = (xmax - xmin).toDouble()

                        // Get label and format label
                        val rawLabel = activity.labels.get(classes.get(index).toInt())
                        val formattedLabel = activity.labelMapping[rawLabel] ?: rawLabel

                        var distance = 0.0
                        var audioFeedback: String
                        var textFeedback: String

                        when (rawLabel) {
                            "entrance_front" -> {
                                distance = activity.distanceFinder(
                                    MainActivity.FOCAL_ENTRANCE_FRONT,
                                    MainActivity.ENTRANCE_WIDTH, objectWidthInFrame)
                            }
                            "entrance_left", "entrance_right" -> {
                                distance = activity.distanceFinder(
                                    MainActivity.FOCAL_ENTRANCE_SIDE,
                                    MainActivity.ENTRANCE_WIDTH, objectWidthInFrame)
                            }

                            "escalator_front" -> {
                                distance = activity.distanceFinder(
                                    MainActivity.FOCAL_ESCALATOR_FRONT,
                                    MainActivity.ESCALATOR_WIDTH, objectWidthInFrame)
                            }
                            "escalator_left", "escalator_right" -> {
                                distance = activity.distanceFinder(
                                    MainActivity.FOCAL_ESCALATOR_SIDE,
                                    MainActivity.ESCALATOR_WIDTH, objectWidthInFrame)
                            }

                            "stair_front" -> {
                                distance = activity.distanceFinder(
                                    MainActivity.FOCAL_STAIR_FRONT,
                                    MainActivity.STAIR_WIDTH, objectWidthInFrame)
                            }
                            "stair_left", "stair_right" -> {
                                distance = activity.distanceFinder(
                                    MainActivity.FOCAL_STAIR_SIDE,
                                    MainActivity.STAIR_WIDTH, objectWidthInFrame)
                            }
                        }

                        val roundedScore = String.format("%.2f", score * 100)
                        val normalizedUnit = activity.normalizeUnit(activity.currentUnit)

                        // Convert distance to the desired unit
                        val convertedDistance = when (normalizedUnit) {
                            "inches" -> distance.roundToInt()
                            "feet" -> (distance / 12.0).roundToInt()
                            "metres" -> String.format("%.2f", distance * INCH_TO_METRE_CONVERSION).toDouble()
                            "centimetres" -> (distance * INCH_TO_CM_CONVERSION).roundToInt()
                            "millimetres" -> (distance * INCH_TO_MM_CONVERSION).roundToInt()
                            else -> distance.roundToInt()
                        }

                        audioFeedback = if (activity.warningEnabled || distance < 50) {
                            if (distance < 50) {
                                // Trigger haptic feedback
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                                    vibrator.vibrate(VibrationEffect.createOneShot(500, VibrationEffect.DEFAULT_AMPLITUDE))
                                } else {
                                    // For devices below API level 26
                                    vibrator.vibrate(500)
                                }
                                "$formattedLabel Be careful!"
                            } else {
                                formattedLabel
                            }
                        } else {
                            "$formattedLabel $convertedDistance $normalizedUnit away"
                        }

                        textFeedback = "$formattedLabel $roundedScore% $convertedDistance $normalizedUnit away"

                        canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), activity.paint)
                        activity.paint.style = Paint.Style.FILL
                        canvas.drawText(rawLabel, locations.get(x+1)*w, locations.get(x)*h, activity.paint)

                        // Check if the text-to-speech engine is currently speaking.
                        val isTTSSpeaking = activity.tts?.isSpeaking ?: false

                        // Check if the text-to-speech engine is currently speaking and if it should speak
                        if (!isTTSSpeaking && activity.speak) {
                            val currentTime = System.currentTimeMillis()
                            if (formattedLabel != activity.lastSpokenLabel || (currentTime - activity.lastSpokenTime) > activity.debounceTime) {
                                activity.tts?.speak(audioFeedback, TextToSpeech.QUEUE_FLUSH, null, null)
                                activity.lastSpokenLabel = formattedLabel
                                activity.lastSpokenTime = currentTime
                            }
                        }

                        Toast.makeText( activity, textFeedback, Toast.LENGTH_SHORT).show()
                    }
                }

                imageView.setImageBitmap(mutable)

            }

        }

        activity.cameraManager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    }
}
