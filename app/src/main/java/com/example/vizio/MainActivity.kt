package com.example.vizio

import android.annotation.SuppressLint
import android.content.Context
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
import android.speech.tts.TextToSpeech
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

    var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)


    private val CAMERA_REQUEST_CODE = 101
    lateinit var cameraManager: CameraManager
    lateinit var cameraDevice: CameraDevice
    lateinit var textureView: TextureView
    lateinit var imageView: ImageView
    lateinit var handler: Handler
    lateinit var bitmap: Bitmap
    lateinit var model: Vizio8
    lateinit var imageProcessor: ImageProcessor
    lateinit var labels: List<String>
    val paint = Paint()

    private var tts: TextToSpeech? = null
    private var lastSpokenText: String? = null
    private var lastSpokenTime: Long = 0
    private val DEBOUNCE_TIME = 7000 // 7 seconds

    companion object {
        const val ENTRANCE_WIDTH = 54.0 //inches
        const val ESCALATOR_WIDTH = 59.0 //inches
        const val STAIR_WIDTH = 86.0 //inches

        const val FOCAL_LENGTH_ESCALATOR = 152.31638418079095
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
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                tts = TextToSpeech(this@MainActivity) { status ->
                    if (status != TextToSpeech.ERROR) {
                        val result = tts?.setLanguage(Locale.US)
                        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                            // Handle error here, notify the user that the language data is missing or not supported
                            Log.e("TTS", "Language not supported or missing data")
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

//                    Log.d("scores", "1 : "
//                            + scores[0].toString() + "," + scores[1].toString() +"," + scores[2].toString() +"," + scores[3].toString()
//                            + scores[4].toString() + "," + scores[5].toString() +"," + scores[6].toString() +"," + scores[7].toString()
//                            + scores[8].toString() + "," + scores[9].toString()
//                    )
//                    Log.d("locations", "2 : " + locations[0].toString())
//                    Log.d("numberOfDetections", "3 : " + numberOfDetections[0].toString())
//                    Log.d("classes", "4 : " + classes[0].toString())

                    if(score > 0.7){
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE

                        val xmin = locations[x + 1] * w
                        val xmax = locations[x + 3] * w

                        val objectWidthInFrame = (xmax - xmin).toDouble()

                        canvas.drawRect(RectF(locations.get(x+1)*w, locations.get(x)*h, locations.get(x+3)*w, locations.get(x+2)*h), paint)
                        paint.style = Paint.Style.FILL

                        val rawLabel = labels.get(classes.get(index).toInt())
                        val formattedLabel = formatLabel(rawLabel)

                        var distance: Double = 0.0
                        var roundedDistance: Int = 0
                        var audioFeedback = ""
                        when (rawLabel) {
                            "entrance_front", "entrance_left", "entrance_right" -> {
//                                distance = distanceFinder(FOCAL_LENGTH_ENTRANCE, ENTRANCE_WIDTH, objectWidthInFrame)
                                audioFeedback = "$formattedLabel"
                            }

                            "escalator_front", "escalator_left", "escalator_right" -> {
                                distance = distanceFinder(FOCAL_LENGTH_ESCALATOR, ESCALATOR_WIDTH, objectWidthInFrame)
                                roundedDistance = distance.roundToInt()
                                audioFeedback = "$formattedLabel $roundedDistance inches away"
                            }

                            "stair_front", "stair_left", "stair_right" -> {
//                                distance = distanceFinder(FOCAL_LENGTH_ENTRANCE, ENTRANCE_WIDTH, objectWidthInFrame)
                                audioFeedback = "$formattedLabel"
                            }
                        }

                        val roundedScore = String.format("%.2f", score * 100)

                        val textFeedback = "$rawLabel $roundedScore% $roundedDistance"

                        canvas.drawText(rawLabel, locations.get(x+1)*w, locations.get(x)*h, paint)

                        // Check if the text-to-speech engine is currently speaking.
                        val isTTSSpeaking = tts?.isSpeaking ?: false

                        if (!isTTSSpeaking) {
                            val currentTime = System.currentTimeMillis()
                            if (audioFeedback != lastSpokenText || (currentTime - lastSpokenTime) > DEBOUNCE_TIME) {
                                tts?.speak(audioFeedback, TextToSpeech.QUEUE_FLUSH, null, null)
                                lastSpokenText = audioFeedback
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

    fun formatLabel(rawLabel: String): String {
        val parts = rawLabel.split('_')
        if (parts.size == 2) {
            val objectPart = parts[0].capitalize()
            val directionPart = parts[1].capitalize()
            return "$objectPart at $directionPart"
        }
        return rawLabel // return original label if it doesn't match expected format
    }

    fun distanceFinder(focalLength: Double, realObjectWidth: Double, widthInFrame: Double): Double {
        return (realObjectWidth * focalLength) / widthInFrame
    }

    @SuppressLint("MissingPermission")
    fun open_camera(){
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

    fun get_permission() {
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