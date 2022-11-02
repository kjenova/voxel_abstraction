function [] = preprocess_urocell()
    startup;
    globals;
    shapenetDir = shapenetDir;
    cachedir = cachedir;

    tsdfDir = fullfile(cachedir, 'shapenet', 'chamferData', 'urocell');

    mkdir(tsdfDir)

    helper('fib1-0-0-0', tsdfDir);
    helper('fib1-1-0-3', tsdfDir);
    helper('fib1-3-2-1', tsdfDir);
    helper('fib1-3-3-0', tsdfDir);
    helper('fib1-4-3-0', tsdfDir);
end

function helper(filename, tsdfDir)
    V = niftiread(fullfile("branched/", strcat(filename, '.nii.gz')));
    mkdir(fullfile(tsdfDir, filename));
    helper2(filename, V, 1, tsdfDir);
    helper2(filename, V, 2, tsdfDir);
end

function helper2(filename, V, label, tsdfDir)
    gridSize = 64;
    numSamples = 10000;
    numShapes = 2000;

    mkdir(fullfile(tsdfDir, filename, num2str(label)));

    CC = bwconncomp(V == label);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [sorted, indices] = sort(numPixels, 'descend');

    pBar = TimedProgressBar(numShapes, 30, 'Time Remaining : ', ' Percentage Completion ', 'Tsdf Extraction Completed.');

    for i = 1:size(sorted, 2)
        if sorted(i) < 2
            break;
        end

        X = zeros(size(V));
        X(CC.PixelIdxList{indices(i)}) = 1;
        cropped = regionprops(X, "FilledImage").FilledImage;
        [FV, ~] = Voxel2mesh(cropped);

        maxSize = max(size(cropped));
        normalizedVertices = (FV.vertices - 0.5) / maxSize;
        standardizedVertices = normalizedVertices - 0.5;

        surfaceSamples = sampling(FV.faces, standardizedVertices, numSamples);

        maxSize = max(size(cropped));
        Volume = imresize3(cropped, gridSize / maxSize);
        padding = gridSize - size(Volume);
        Volume = padarray(Volume, padding, 0, 'post');
        if ~isequal(size(Volume), [gridSize gridSize gridSize])
            disp("!!!!!!!!");
            disp(i);
        end

        stepRange = -0.5 + 1 / (2 * gridSize) :1 / gridSize : 0.5 - 1 / (2 * gridSize);
        [Xp,Yp,Zp] = ndgrid(stepRange, stepRange, stepRange);
        queryPoints = [Xp(:), Yp(:), Zp(:)];

        [tsdfPoints,~,closestPoints] = point_mesh_squared_distance(queryPoints, standardizedVertices, FV.faces);
        tsdfPoints = sqrt(tsdfPoints);
        tsdfGrid = reshape(tsdfPoints, size(Xp));
        tsdfGrid = abs(tsdfGrid) .* (1 - 2 * Volume);
        closestPointsGrid = reshape(closestPoints, [size(Xp), 3]);

        tsdfFile = fullfile(tsdfDir, filename, num2str(label), strcat(num2str(i), '.mat'));
        savefunc(tsdfFile, tsdfGrid, Volume, closestPointsGrid, surfaceSamples(:, 1:3), standardizedVertices, FV.faces, surfaceSamples(:, 4:6));
        pBar.progress();
    end

    pBar.stop();
end

function savefunc(tsdfFile, tsdf, Volume, closestPoints, surfaceSamples, vertices, faces, normals)
    save(tsdfFile, 'tsdf', 'Volume', 'closestPoints', 'surfaceSamples', 'vertices', 'faces', 'normals');
end

function [finalSamples] = sampling(faces, vertices, numSamples)
    samples = basic_sampling(faces, vertices, numSamples);

    finalSamples = zeros(numSamples, 6);
    finalSamples(:, 1:3) = samples(:, 1:3);

    faceCenters = face_centers(faces, vertices);

    k = 6;
    D = pdist2(samples(:, 1:3), faceCenters(:, 1:3));
    for sId = 1:numSamples
        normal = zeros(1, 3);
        [B, I] = mink(D(sId, :), k);
        B = 1 ./ B;
        B = B ./ sum(B);
        for j = 1:k
            normal = normal + B(j) * faceCenters(I(j), 4:6);
        end
        normal = normal / norm(normal);
        finalSamples(sId, 4:6) = normal;
    end
end

function [samples] = basic_sampling(faces, vertices, numSamples)
    samples = zeros(numSamples, 6);
    faceIds = datasample(1:size(faces, 1), numSamples);
    paras = rand(2, numSamples);

    for sId = 1:numSamples
        faceId = faceIds(sId);
        p1 = vertices(faces(faceId, 1), :);
        p2 = vertices(faces(faceId, 2), :);
        p3 = vertices(faces(faceId, 3), :);
        
        r1 = paras(1, sId);
        r2 = paras(2, sId);
        t1 = 1-sqrt(r1);
        t2 = sqrt(r1)*(1-r2);
        t3 = sqrt(r1)*r2;
        samples(sId, 1:3) = t1*p1 + t2*p2 + t3*p3;

        c = cross(p2 - p1, p3 - p1);
        samples(sId, 4:6) = c / norm(c);
    end
end

function [centers] = face_centers(faces, vertices)
    centers = zeros(size(faces, 1), 6);
    for faceId = 1:size(faces, 1)
        p1 = vertices(faces(faceId, 1), :);
        p2 = vertices(faces(faceId, 2), :);
        p3 = vertices(faces(faceId, 3), :);

        centers(faceId, 1:3) = (p1 + p2 + p3) / 3;
        c = cross(p2 - p1, p3 - p1);
        centers(faceId, 4:6) = c / norm(c);
    end
end
